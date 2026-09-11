"""Export validation learning curves from recorded logs; no test-set selection."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.dpi': 180})
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.6), constrained_layout=True)
    for column, task in enumerate((1, 2)):
        path = args.workspace / f'runs/task{task}-act/metrics.jsonl'
        records = [json.loads(line) for line in path.read_text().splitlines()]
        validation = [record for record in records if 'validation' in record]
        best = min(validation, key=lambda record: record['validation']['normalized_mae'])
        steps = [record['step'] for record in validation]
        for row, (metric, baseline, label) in enumerate((
            ('normalized_mae', 'train_mean_baseline_normalized_mae', 'Normalized action MAE'),
            ('model_first_joint_action_mae_rad', 'joint_persistence_first_action_mae_rad',
             'First joint-target MAE (rad)'),
        )):
            ax = axes[row, column]
            ax.plot(steps, [r['validation'][metric] for r in validation], color='#086788',
                    marker='.', label='ACT on validation episodes')
            ax.axhline(validation[0]['validation'][baseline], color='#9b5c13', linestyle='--',
                       linewidth=1.2, label='Training mean' if row == 0 else 'Measured-joint persistence')
            ax.scatter([best['step']], [best['validation'][metric]], color='#c44e52',
                       zorder=3, s=45, label='Selected checkpoint')
            ax.set(xlabel='Optimizer step', ylabel=label)
            ax.grid(alpha=.18)
            ax.legend(fontsize=8)
        axes[0, column].set_title(f'Task {task}: full-release training')
    fig.suptitle('AirSign Phase II — offline validation, not physical task scores', fontsize=13)
    args.output.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'pdf'):
        fig.savefig(args.output / f'validation-curves.{extension}')
    plt.close(fig)
    print(json.dumps({'figure': str(args.output / 'validation-curves.png')}))


if __name__ == '__main__':
    main()
