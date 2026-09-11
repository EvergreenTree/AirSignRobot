"""Run pinned detector/segmenter weights on real recorded RGB, without invented depth."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from ebim_phase2.download import sha256
from ebim_phase2.vision import RGBDObjectPerception


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',type=Path,required=True)
    p.add_argument('--images',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--wait-for-download',action='store_true')
    args=p.parse_args()
    detector=args.models/'grounding-dino-tiny';segmenter=args.models/'sam2.1-hiera-tiny'
    if args.wait_for_download:
        print('Waiting for both immutable, hash-verified model manifests',flush=True)
        while not all((folder/'download_manifest.json').exists() for folder in (detector,segmenter)):time.sleep(10)
    torch.set_num_threads(4)
    perception=RGBDObjectPerception(detector_dir=detector,segmenter_dir=segmenter,device='cuda')
    prompts={'gray_strip':'a gray rectangular strip','memory_module':'a green computer memory module',
             'red_region':'a red rectangular strip'}
    args.output.mkdir(parents=True,exist_ok=True)
    summaries=[]
    for path in args.images:
        rgb=np.asarray(Image.open(path).convert('RGB'))
        report,masks=perception.infer(rgb,None,None,prompts,timestamp=0.,camera_id=path.stem)
        report['input_sha256']=sha256(path)
        report['validation_scope']='released Task2 RGB only; no depth, pose, mask ground truth, face or success labels'
        assert all(item['geometry'] is None for item in report['objects'])
        (args.output/f'{path.stem}.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        np.savez_compressed(args.output/f'{path.stem}-masks.npz',**masks)
        summaries.append({'image':str(path),'objects':len(report['objects']),
                          'missing_labels':report['missing_labels'],'elapsed_seconds':report['elapsed_seconds']})
        print(json.dumps(summaries[-1]),flush=True)
    result={'team':'AirSign','models':perception.models,'transformers':perception.transformers_version,
            'weight_loading':perception.loading_report,
            'torch':torch.__version__,'images':summaries,'physical_validation':None}
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
