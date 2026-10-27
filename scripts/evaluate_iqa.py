#!/usr/bin/env python3
"""Evaluate the public metric using explicit FR-IQA image/MOS CSV rows.

Required columns: ref_path, dist_path, raw_target. Reference features are cached;
the aggregation formula is identical to MLCLIPSim.forward. No scores are reused.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time
import warnings

import numpy as np
from PIL import Image
from scipy import optimize, stats
import torch
from torchvision.transforms.functional import to_tensor

from mlclipsim import load_metric
from mlclipsim.metric import _l2_normalize


def logistic4(x, b1, b2, b3, b4):
    return (b1-b2)/(1+np.exp(-(x-b3)/(abs(b4)+1e-12)))+b2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs-csv', type=Path, required=True)
    p.add_argument('--score-type', choices=['mos', 'dmos'], required=True)
    p.add_argument('--clip-checkpoint', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--max-items', type=int)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    metric = load_metric(args.checkpoint, device='cpu', clip_checkpoint=args.clip_checkpoint)
    rows = list(csv.DictReader(args.pairs_csv.open()))
    if args.max_items:
        rows = rows[:args.max_items]
    refs, records = {}, []
    weights, gate = metric._layer_weights(torch.device('cpu')), torch.sigmoid(metric.global_weight/metric.tau_gate)
    started = time.monotonic()

    def image(path):
        with Image.open(path) as im:
            tensor = to_tensor(im.convert('RGB')).unsqueeze(0)
        return metric._preprocess(tensor)

    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start+args.batch_size]
            for row in batch:
                path = row['ref_path']
                if path not in refs:
                    emb, feat = metric._encode(image(path), detach=True)
                    refs[path] = (_l2_normalize(emb), [_l2_normalize(x) for x in feat])
            emb, feat = metric._encode(torch.cat([image(r['dist_path']) for r in batch]), detach=True)
            global_ref = torch.cat([refs[r['ref_path']][0] for r in batch])
            global_score = (global_ref*_l2_normalize(emb)).sum(-1)
            token_score = torch.zeros(len(batch))
            for j, candidate in enumerate(feat):
                reference = torch.cat([refs[r['ref_path']][1][j] for r in batch])
                sims = (reference[:, 1:]*_l2_normalize(candidate)[:, 1:]).sum(-1).mean(1)
                token_score += weights[j]*sims
            scores = gate*token_score+(1-gate)*global_score
            for row, score in zip(batch, scores.tolist()):
                records.append({'ref_path':row['ref_path'], 'dist_path':row['dist_path'],
                                'target':float(row['raw_target']), 'score':score})
            if start % 256 == 0 or len(records)==len(rows):
                print(json.dumps({'images':len(records),'total':len(rows),
                                  'seconds':time.monotonic()-started}),flush=True)
    pred = np.asarray([r['score'] for r in records]);target = np.asarray([r['target'] for r in records])
    if args.score_type=='dmos':target=-target
    p0 = [target.max(),target.min(),pred.mean(),pred.std()+1e-6]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            fit,_=optimize.curve_fit(logistic4,pred,target,p0=p0,maxfev=20000)
            fitted=logistic4(pred,*fit);fit_status='four_parameter_logistic'
        except (RuntimeError, ValueError):
            fitted=pred;fit_status='fit_failed_raw_pearson'
    result={'count':len(records),'torch':torch.__version__,'device':'cpu',
            'score_type':args.score_type,'checkpoint':str(args.checkpoint) if args.checkpoint else 'bundled',
            'srcc':float(stats.spearmanr(pred,target).statistic),
            'krcc':float(stats.kendalltau(pred,target).statistic),
            'plcc':float(stats.pearsonr(fitted,target).statistic),'plcc_protocol':fit_status,
            'raw_plcc':float(stats.pearsonr(pred,target).statistic),
            'seconds':time.monotonic()-started,'per_image':records}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='per_image'}),flush=True)


if __name__=='__main__':
    main()
