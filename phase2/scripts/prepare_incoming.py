"""Prepare complete episodes during downloads; publish a training manifest only after full verification."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

from ebim_phase2.data import prepare, prepare_episode
from ebim_phase2.schema import CAMERAS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--image-size',type=int,default=160)
    p.add_argument('--stride',type=int,default=2)
    p.add_argument('--max-unaligned-fraction',type=float,default=.05)
    args=p.parse_args()
    pending={};seen=set();failures=[]
    with ProcessPoolExecutor(args.workers) as pool:
        while True:
            for future, key in list(pending.items()):
                if future.done():
                    try:
                        record=future.result()
                        print(json.dumps({'prepared':key,'observations':record['observations']}),flush=True)
                    except Exception as error:
                        failures.append({'episode':key,'error':repr(error)})
                        print(json.dumps(failures[-1]),flush=True)
                    del pending[future]
            for raw in args.raw:
                if not (raw/'meta/info.json').exists():continue
                info=json.loads((raw/'meta/info.json').read_text())
                source=json.loads((raw/'source.json').read_text())
                originals=info.get('creation_metadata',{}).get('source_episodes',[])
                for path in sorted(raw.glob('data/**/*.parquet')):
                    index=int(path.stem.split('_')[-1]);key=f"{source['release']}_{index:06d}"
                    if key in seen or len(pending)>=args.workers*2:continue
                    if not all((raw/info['video_path'].format(episode_chunk=index//info['chunks_size'],
                                episode_index=index,video_key=c)).exists() for c in CAMERAS):continue
                    job={'raw':str(raw.resolve()),'output':str(args.output.resolve()),'id':key,'task':source['task'],
                         'revision':source['revision'],'source_episode':originals[index] if index<len(originals) else f"{source['repo_id']}:{index}",
                         'site':'munich' if source['task']==2 else 'not_provided','episode_index':index,
                         'parquet':str(path.relative_to(raw)),
                         'max_unaligned_fraction':args.max_unaligned_fraction,
                         'settings':{'image_size':args.image_size,'stride':args.stride,'max_skew_s':.075}}
                    pending[pool.submit(prepare_episode,job)]=key;seen.add(key)
            if all((r/'download_manifest.json').exists() for r in args.raw) and not pending:break
            time.sleep(5)
    if failures:
        raise RuntimeError(f'Preprocessing audit failures: {failures}')
    prepare(args.raw,args.output,args.image_size,args.stride,args.workers,.075,42,args.max_unaligned_fraction)


if __name__=='__main__':main()
