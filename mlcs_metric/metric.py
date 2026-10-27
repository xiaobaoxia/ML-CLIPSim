import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from torch import amp

# CLIP official mean / std
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)


def l2norm(x: torch.Tensor, eps: float = 1e-6):
    x = x.float()
    return x / (x.norm(dim=-1, keepdim=True) + eps)


class CLIPMultiLayerMetric(nn.Module):
    """
    CLIP multi-layer token similarity score (higher is better).

    This version adds:
      1) Grouped layer weights (early/mid/late) to avoid one-hot single-layer collapse.
      2) Attention-weighted patch similarity using a semantic proxy:
         CLS-to-patch affinity on REF tokens (no CLIP source patch required).
      3) Bounded convex gate between token score and global CLIP cosine.

    Intended usage:
      - Train metric on machine-IQA: learn group_weight/global_weight/(optional)attn_mix_logit
      - Freeze metric and train codec with loss = -(score.mean()) or logistic loss
    """

    def __init__(
        self,
        device="cuda",
        layers=None,
        use_patches=True,
        use_cls=False,
        input_size=224,
        patch_mining="mean",        # "mean" or "bottomk" (cosine only)
        bottomk_ratio=0.25,
        tau_group=2.0,              # temperature for group weights
        tau_gate=2.0,               # temperature for gate
        # --- grouped weighting ---
        group_mode="fixed12",       # "fixed12" or "auto3"
        # fixed12 uses (0-3)(4-7)(8-11) assuming 12 layers selected (ViT-B/16)
        # auto3 splits selected layers into 3 equal contiguous groups
        # --- attention-weighted patch similarity ---
        use_attn_weight=True,
        attn_from_layer=-1,         # which selected layer to derive weights from (default last selected)
        attn_tau=1.0,
        attn_gamma=1.0,
        attn_mix=0.6,               # lambda in [0,1]
        learn_attn_mix=True,       # learn lambda but keep bounded
        attn_mix_bounds=(0.2, 0.8),
        # --- optional minimum group weight to prevent group collapse ---
        group_min=0.00,             # e.g. 0.05 to force each group >=5%
        clip_checkpoint=None,       # optional local OpenAI CLIP backbone
    ):
        super().__init__()

        self.device = device
        self.use_patches = use_patches
        self.use_cls = use_cls
        self.input_size = input_size

        self.patch_mining = patch_mining
        self.bottomk_ratio = float(bottomk_ratio)
        self.tau_group = float(tau_group)
        self.tau_gate = float(tau_gate)

        self.group_mode = group_mode
        self.group_min = float(group_min)

        self.use_attn_weight = bool(use_attn_weight)
        self.attn_from_layer = int(attn_from_layer)
        self.attn_tau = float(attn_tau)
        self.attn_gamma = float(attn_gamma)
        self.attn_mix = float(attn_mix)
        self.learn_attn_mix = bool(learn_attn_mix)
        self.attn_mix_bounds = tuple(attn_mix_bounds)

        # CLIP backbone (frozen)
        self.clip_model, _ = clip.load(
            str(clip_checkpoint) if clip_checkpoint is not None else "ViT-B/16",
            device=device, jit=False,
        )
        self.clip_model.float()
        self.clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad = False

        # Layer selection
        self.num_layers = len(self.clip_model.visual.transformer.resblocks)
        if layers is None:
            self.layers = list(range(self.num_layers))
        else:
            self.layers = list(layers)
        self.num_used_layers = len(self.layers)

        # Group definition
        self.layer_groups = self._make_groups()

        # learnable group weights + gate
        self.group_weight = nn.Parameter(torch.zeros(len(self.layer_groups)))
        self.global_weight = nn.Parameter(torch.zeros(1))

        # optional learnable lambda for attention mixing (bounded)
        if self.learn_attn_mix:
            init = float(attn_mix)
            init = min(max(init, 1e-3), 1 - 1e-3)
            logit = torch.log(torch.tensor(init / (1 - init)))
            self.attn_mix_logit = nn.Parameter(logit.view(1))
        else:
            self.attn_mix_logit = None

        # CLIP mean/std buffers
        self.register_buffer(
            "clip_mean",
            torch.tensor(CLIP_MEAN).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "clip_std",
            torch.tensor(CLIP_STD).view(1, 3, 1, 1),
            persistent=False,
        )

        # Hook buffer
        self._feat_buffer = [None] * self.num_layers
        self._register_hooks()

        self.to(device)

    def _make_groups(self):
        """
        Returns list of groups, each group is a list of indices into self.layers (NOT global layer ids).
        """
        L = self.num_used_layers
        if self.group_mode == "auto3":
            # split selected layers into 3 contiguous groups by index
            a = list(range(L))
            g1 = a[: max(1, L // 3)]
            g2 = a[max(1, L // 3): max(2, 2 * L // 3)]
            g3 = a[max(2, 2 * L // 3):]
            groups = [g for g in [g1, g2, g3] if len(g) > 0]
            return groups

        # fixed12 mode: group by global layer id ranges (0-3,4-7,8-11) intersect with selected layers
        # (works best for ViT-B/16 with 12 layers)
        fixed_ranges = [(0, 3), (4, 7), (8, 11)]
        groups = []
        for a, b in fixed_ranges:
            idxs = [k for k, lid in enumerate(self.layers) if a <= lid <= b]
            if len(idxs) > 0:
                groups.append(idxs)
        # Distribute layers across three groups for non-default layer counts.
        if len(groups) == 0:
            self.group_mode = "auto3"
            return self._make_groups()
        return groups

    # Hook registration
    def _register_hooks(self):
        def make_hook(layer_idx):
            def hook_fn(_, __, output):
                if output is not None and output.dim() == 3:
                    self._feat_buffer[layer_idx] = output.permute(1, 0, 2).contiguous()
            return hook_fn

        for i, blk in enumerate(self.clip_model.visual.transformer.resblocks):
            blk.register_forward_hook(make_hook(i))

    # Preprocess
    def _prep(self, x):
        x = x.float()
        if x.shape[-2:] != (self.input_size, self.input_size):
            x = F.interpolate(
                x,
                size=(self.input_size, self.input_size),
                mode="bicubic",
                align_corners=False,
            )
        x = x.clamp(0, 1)
        x = (x - self.clip_mean) / self.clip_std
        return x

    # One CLIP forward
    def _forward_clip(self, x, detach):
        for i in range(self.num_layers):
            self._feat_buffer[i] = None

        with amp.autocast("cuda", enabled=False):
            x = x.float()
            if detach:
                with torch.no_grad():
                    f = self.clip_model.encode_image(x)
            else:
                f = self.clip_model.encode_image(x)

        feats = [self._feat_buffer[i] for i in self.layers]
        return f, feats

    def _build_attn_weights_from_ref(self, feats0):
        """
        feats0: list of [B, T, D] tokens from selected layers (ref branch)
        return: attn_w [B, P], sum=1
        """
        # choose which layer (selected index)
        if self.attn_from_layer < 0:
            idx = len(feats0) - 1
        else:
            if self.attn_from_layer in self.layers:
                idx = self.layers.index(self.attn_from_layer)
            else:
                idx = len(feats0) - 1

        tok = feats0[idx].float()  # [B, T, D]

        # (1) make tokens more "final-space-like"
        # In openai/clip ViT, ln_post exists and is used before projection.
        if hasattr(self.clip_model.visual, "ln_post"):
            tok = self.clip_model.visual.ln_post(tok)

        # normalize tokens so dot = cosine
        tok = l2norm(tok)
        cls = tok[:, 0:1, :]  # [B,1,D]
        patches = tok[:, 1:, :]  # [B,P,D]

        affinity = (cls * patches).sum(dim=-1)  # [B,P]

        # (2) widen dynamic range: standardize per-sample
        aff_mean = affinity.mean(dim=1, keepdim=True)
        aff_std = affinity.std(dim=1, keepdim=True).clamp_min(1e-3)
        z = (affinity - aff_mean) / aff_std

        # (3) sharper softmax
        tau = getattr(self, "attn_tau", 0.3)  # suggest start 0.3
        a = torch.softmax(z / max(tau, 1e-6), dim=1)

        # optional gamma sharpening
        gamma = getattr(self, "attn_gamma", 2.0)  # suggest start 2.0
        if abs(gamma - 1.0) > 1e-6:
            a = a.clamp_min(1e-8).pow(gamma)
            a = a / (a.sum(dim=1, keepdim=True) + 1e-6)

        return a.detach(),z
    def _group_to_layer_weights(self, device):
        """
        Returns w_layer [L] from group weights.
        Group weights sum to 1; each group weight distributed uniformly to its member layers.
        """
        G = len(self.layer_groups)
        w_g = torch.softmax(self.group_weight / self.tau_group, dim=0)  # [G]

        if self.group_min > 0:
            # force each group >= group_min
            m = self.group_min
            if m * G >= 1.0:
                raise ValueError("group_min too large; must satisfy group_min*G < 1.")
            w_g = w_g * (1.0 - m * G) + m

        L = self.num_used_layers
        w_layer = torch.zeros(L, device=device)

        for gi, idxs in enumerate(self.layer_groups):
            if len(idxs) == 0:
                continue
            per = w_g[gi] / len(idxs)
            for k in idxs:
                w_layer[k] = per

        # renormalize in case some layers not covered
        w_layer = w_layer / (w_layer.sum() + 1e-6)
        return w_layer, w_g

    # Forward
    def forward(self, ref, dist, return_debug=False):
        ref = self._prep(ref)
        dist = self._prep(dist)

        f0, feats0 = self._forward_clip(ref, detach=True)
        f1, feats1 = self._forward_clip(dist, detach=False)

        B = ref.shape[0]
        score_layers = torch.zeros(B, device=ref.device)

        # grouped layer weights
        w_layer, w_g = self._group_to_layer_weights(ref.device)

        # attention weights + mix lambda
        attn_w = None
        lam = self.attn_mix
        if self.use_attn_weight and self.use_patches:
            attn_w,z = self._build_attn_weights_from_ref(feats0)  # [B,P]

            if self.learn_attn_mix:
                lo, hi = self.attn_mix_bounds
                lam = lo + (hi - lo) * torch.sigmoid(self.attn_mix_logit)
                lam = lam.squeeze()
            else:
                lam = float(lam)

        per_layer_s = []
        sim_patch_list = [] if return_debug else None
        for j in range(self.num_used_layers):
            fr_raw = feats0[j].float()
            fd_raw = feats1[j].float()
            fr = l2norm(fr_raw)
            fd = l2norm(fd_raw)
            s = torch.zeros(B, device=ref.device)
            if self.use_cls:
                s = s + (fr[:, 0] * fd[:, 0]).sum(dim=-1)
            if self.use_patches:
                sim_patch = (fr[:, 1:] * fd[:, 1:]).sum(dim=-1)  # [B,P]
                # uniform aggregation
                if self.patch_mining == "mean":
                    s_u = sim_patch.mean(dim=1)
                else:
                    k = max(1, int(sim_patch.shape[1] * self.bottomk_ratio))
                    s_u = torch.topk(sim_patch, k=k, dim=1, largest=False).values.mean(dim=1)
                if attn_w is not None:
                    s_a = (attn_w * sim_patch).sum(dim=1)
                    s_patch = (1.0 - lam) * s_u + lam * s_a
                else:
                    s_patch = s_u
                s = s + s_patch

                if return_debug:
                    sim_patch_list.append(sim_patch.detach())
            score_layers = score_layers + w_layer[j] * s
            if return_debug:
                per_layer_s.append(s.detach())

        # global CLIP cosine
        f0n = l2norm(f0.float())
        f1n = l2norm(f1.float())
        clip_cos = (f0n.detach() * f1n).sum(dim=-1)

        # bounded convex gate between token score and global score
        g = torch.sigmoid(self.global_weight / self.tau_gate)  # scalar
        score = g * score_layers + (1.0 - g) * clip_cos

        if return_debug:
            # entropies to monitor collapse
            group_entropy = -(w_g * (w_g + 1e-8).log()).sum()
            gate_entropy = -(g * torch.log(g + 1e-8) + (1 - g) * torch.log(1 - g + 1e-8))
            dbg = {
                "g": g.detach(),
                "w_g": w_g.detach(),
                "w_layer": w_layer.detach(),
                "group_entropy": group_entropy.detach(),
                "gate_entropy": gate_entropy.detach(),
                "score_layers": score_layers.detach(),
                "clip_cos": clip_cos.detach(),
            }
            if attn_w is not None:
                dbg["attn_mix_lam"] = torch.tensor(float(lam), device=ref.device).detach()
                dbg["attn_w_max"] = attn_w.max(dim=1).values.detach()  # [B]
                dbg["attn_w"] = attn_w.detach()  # [B, P]
                dbg["z"] = z.detach()  # [B, P]
            if sim_patch_list is not None and len(sim_patch_list) > 0:
                dbg["sim_patch"] = torch.stack(sim_patch_list, dim=0).detach()  # [L, B, P]
            if len(per_layer_s) > 0:
                dbg["per_layer_s_mean"] = torch.stack(per_layer_s, dim=0).mean(dim=1)

            return score, dbg

        return score

    @torch.no_grad()
    def score(self, ref, dist):
        return self.forward(ref, dist, return_debug=False)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    metric = CLIPMultiLayerMetric(
        device=device,
        layers=None,               # use all 12 layers
        use_patches=True,
        use_cls=False,
        patch_mining="mean",
        bottomk_ratio=0.25,
        tau_group=3.0,
        tau_gate=2.0,
        group_mode="fixed12",      # (0-3)(4-7)(8-11)
        group_min=0.00,            # set 0.05 if you want to force each group >=5%
        use_attn_weight=True,
        attn_from_layer=-1,
        attn_tau=1.0,
        attn_gamma=1.0,
        attn_mix=0.6,
        learn_attn_mix=False,
        attn_mix_bounds=(0.2, 0.8),
    ).to(device)

    ref = torch.rand(2, 3, 256, 256, device=device)
    dist = torch.rand(2, 3, 256, 256, device=device)

    out, dbg = metric(ref, dist, return_debug=True)
    print("score:", out)
    print("g:", dbg["g"].item())
    print("w_g:", dbg["w_g"])
    print("w_layer:", dbg["w_layer"])
    print("group_entropy:", dbg["group_entropy"].item())
    print("gate_entropy:", dbg["gate_entropy"].item())
    if "attn_mix_lam" in dbg:
        print("attn_mix_lam:", dbg["attn_mix_lam"].item())
        print("attn_w_max (per-sample):", dbg["attn_w_max"])
