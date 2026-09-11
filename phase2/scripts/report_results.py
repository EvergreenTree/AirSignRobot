"""Build a reviewable offline-results report from completed, full-release runs."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

from ebim_phase2.download import sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    base=args.workspace.resolve()
    tasks=[]
    for task in (1,2):
        dataset=base/f'data/task{task}-full/manifest.json'
        manifest=json.loads(dataset.read_text())
        if manifest['coverage']!='full_official_release':raise ValueError('Final report requires the full official release')
        run=json.loads((base/f'runs/task{task}-act/run.json').read_text())
        test=json.loads((base/f'runs/task{task}-test/test_metrics.json').read_text())
        export=json.loads((base/f'exports/task{task}/manifest.json').read_text())
        records=[json.loads(line) for line in (base/f'runs/task{task}-act/metrics.jsonl').read_text().splitlines()]
        validation=[r for r in records if 'validation' in r]
        best=min(validation,key=lambda r:r['validation']['normalized_mae'])
        if test['checkpoint_sha256']!=sha256(base/f'runs/task{task}-act/best.pt'):
            raise ValueError('Reserved test metrics refer to a different checkpoint')
        if run['dataset_sha256']!=sha256(dataset):raise ValueError('Training dataset manifest changed')
        if (export['training_checkpoint_sha256'] != test['checkpoint_sha256']
                or export['dataset_sha256'] != run['dataset_sha256']
                or export['training_step'] != best['step']
                or export['checkpoint_sha256'] != sha256(base/f'exports/task{task}/policy.pt')):
            raise ValueError('Export does not match the selected, tested checkpoint')
        if test['samples'] != sum(e['observations'] for e in manifest['episodes'] if e['split']=='test'):
            raise ValueError('Reserved test report must cover every prepared test observation')
        changed_source = [name for name, digest in run['source_hashes'].items()
                          if sha256(base/'source/ebim_phase2'/name) != digest]
        if set(changed_source) & {'model.py', 'train.py', 'data.py', 'schema.py'}:
            raise ValueError('Training or data source changed after the recorded run')
        tasks.append({'task':task,'dataset_sha256':run['dataset_sha256'],
                      'source_revisions':manifest['source_revisions'],
                      'episode_split':dict(Counter(e['split'] for e in manifest['episodes'])),
                      'robot_frames':sum(e['frames'] for e in manifest['episodes']),
                      'prepared_observations':sum(e['observations'] for e in manifest['episodes']),
                      'dropped_unaligned_observations':sum(e['dropped_unaligned_observations'] for e in manifest['episodes']),
                      'alignment_policy':manifest.get('alignment_policy'),
                      'training_source_hashes':run['source_hashes'],
                      'non_training_modules_changed_since_launch':changed_source,
                      'best_step':best['step'],'validation':best['validation'],'test':test,'export':export,
                      'last_step':max(r['step'] for r in records)})
    args.output.mkdir(parents=True,exist_ok=True)
    for task in (1, 2):
        for source, name in ((base/f'data/task{task}-full/manifest.json', 'dataset-manifest.json'),
                             (base/f'runs/task{task}-act/run.json', 'run.json'),
                             (base/f'runs/task{task}-act/metrics.jsonl', 'training-metrics.jsonl')):
            shutil.copyfile(source, args.output/f'task{task}-{name}')
    diagnosis = base/'runs/task1-joint5-data-diagnosis.json'
    if diagnosis.exists():
        shutil.copyfile(diagnosis, args.output/diagnosis.name)
    report={'team':'AirSign','phase':2,'tasks':tasks,'physical_validation':None,
            'task3':{'implementation':'measured-feedback skill program','demonstration_checkpoint':None,'physical_validation':None}}
    (args.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# AirSign Phase II offline results','',
           'These results measure held-out native action prediction. They are not competition scores or physical success rates.',
           '', '| Task | Train / validation / test episodes | Selected step | Validation normalized MAE | Test normalized MAE | Test mean baseline |',
           '|---|---|---:|---:|---:|---:|']
    for item in tasks:
        counts=item['episode_split'];t=item['test']
        lines.append(f"| {item['task']} | {counts['train']} / {counts['validation']} / {counts['test']} | {item['best_step']} | {item['validation']['normalized_mae']:.6f} | {t['normalized_mae']:.6f} | {t['train_mean_baseline_normalized_mae']:.6f} |")
    lines+=['','Whole episodes were split before training, with training-only normalization. Checkpoint selection used validation data; the reserved test split did not select a model. Splits are within the supplied releases and do not establish cross-site generalization.',
            '', '| Task | Test first joint-target MAE (native radians) | Worst joint channel MAE | Measured-joint persistence baseline | Test observations |',
            '|---|---:|---:|---:|---:|']
    for item in tasks:
        t=item['test'];worst=max(t['first_action_native_mae_by_dimension'][i] for i in (*range(7),*range(8,15)))
        lines.append(f"| {item['task']} | {t['model_first_joint_action_mae_rad']:.6f} | {worst:.6f} | {t['joint_persistence_first_action_mae_rad']:.6f} | {t['samples']} |")
    lines+=['','The joint baseline compares measured joints with recorded GELLO targets. It is an offline reference, not a measurement of physical servo accuracy. Per-dimension native errors, full source revisions, data hashes and export provenance are in `results.json`.',
            '', 'Task 1 has a material data-contract issue: part 2 episodes 5–8 hold the recorded right-joint-5 GELLO target at −2.8763 rad while the measured joint stays near −1.18 rad. Episodes 5 and 7 are in test; 6 and 8 are in training. The test first-action error for this channel is 0.4875 rad. The recording alone does not identify whether this is an unexecuted/disabled-controller target or another interface convention. This diagnosis was made after the fixed-model test; no episodes were removed, labels rewritten, or model retuned. See `task1-joint5-data-diagnosis.json`.',
            '', 'Normalized MAE averages every native action dimension, including constant channels; its scale differs between tasks. Task 1 includes recorded base-command feedback among its inputs, so low base-action error does not establish navigation. Joint errors above include both arms, even when one arm moves little.',
            '', '![Validation learning curves](validation-curves.png)',
            '', 'The accompanying `software-validation.json` records policy/HTTP checks, fixed-checkpoint image diagnostics, wheel installation, pretrained perception loading and the separate MuJoCo actuator probe. Dataset manifests, run configurations and complete training metrics accompany this report.',
            '', 'Task 1 needs the assigned-route verifier and closed-loop evaluation of new routes; the ACT network itself has no route-instruction input. Task 2 needs pad-face and contact validation. Task 3 has a fresh feedback controller and optional object perception, with no demonstration checkpoint or measured task score.',
            '', 'All physical paths still need the actual Phase II transport, site calibration, task-specific grasp/outcome perception, and organizer or robot trials. No real deployment is claimed.']
    (args.output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'report':str(args.output/'RESULTS.md'),'tasks':[t['task'] for t in tasks]}))


if __name__=='__main__':main()
