"""Validation-only image counterfactuals for a fixed checkpoint; no model selection on test."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ebim_phase2.data import EpisodeDataset
from ebim_phase2.download import sha256
from ebim_phase2.model import ACT, ACTConfig
from ebim_phase2.train import evaluate


class CounterfactualImages(Dataset):
    def __init__(self,data,indices,mode):self.data,self.indices,self.mode=data,indices,mode
    def __len__(self):return len(self.indices)
    def __getitem__(self,index):
        i=int(self.indices[index]);item=self.data[i]
        if self.mode=='gray':item['images']=np.full_like(item['images'],127)
        elif self.mode=='mismatched':item['images']=self.data[(i+len(self.data)//2)%len(self.data)]['images']
        return item


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--samples',type=int,default=1024)
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if checkpoint['dataset_sha256']!=sha256(args.dataset/'manifest.json'):raise ValueError('Checkpoint and dataset differ')
    config=ACTConfig(**checkpoint['model_config'])
    model=ACT(config,pretrained=False).cuda().eval();model.load_state_dict(checkpoint['model'])
    dataset=EpisodeDataset(args.dataset,checkpoint['task'],'validation',config.chunk_size,checkpoint['stats'])
    indices=np.linspace(0,len(dataset)-1,min(args.samples,len(dataset)),dtype=int)
    metrics={}
    for mode in ('recorded','gray','mismatched'):
        loader=DataLoader(CounterfactualImages(dataset,indices,mode),batch_size=64,num_workers=args.workers,
                          multiprocessing_context='spawn' if args.workers else None,pin_memory=True)
        metrics[mode]=evaluate(model,loader,torch.device('cuda'),checkpoint['stats'],True)
        print(json.dumps({'mode':mode,'normalized_mae':metrics[mode]['normalized_mae']}),flush=True)
    report={'task':checkpoint['task'],'checkpoint_sha256':sha256(args.checkpoint),'split':'validation',
            'validation_episodes':len(dataset.episodes),'dataset_coverage':dataset.manifest.get('coverage'),
            'metrics':metrics,'scope':'fixed-policy counterfactual image sensitivity; neither a task score nor physical validation'}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
