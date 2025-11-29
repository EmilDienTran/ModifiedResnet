import pandas as pd
import matplotlib.pyplot as plt
import os

'''
This is for visualisation of metrics for the report; it was created by Claude Sonnet 4.5
'''


# Define models and base metrics
base_metrics = ['Acc_test', 'Acc_train', 'F1', 'Loss_test', 'Precision', 'Recall']
models = ['Resnet18', 'ModifiedResNetAttention', 'ModifiedResnetLayered']

csv_dir = 'Metrics/CSVFiles/CIFAR100'  # Change if needed

for model in models:
    # Determine metrics and grid size for this model
    if model == 'ModifiedResnetLayered':
        metrics = base_metrics + ['Weight_attn', 'Weight_conv']
        fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    else:
        metrics = base_metrics
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes = axes.flatten()

    fig.suptitle(f'{model} CIFAR100', fontsize=16, fontweight='bold')

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        filename = f'{metric}_{model}_best.csv'
        filepath = os.path.join(csv_dir, filename)

        try:
            df = pd.read_csv(filepath)
            # Plot columns 1 and 2 (skipping index/step column)
            ax.plot(df.iloc[:, 1], df.iloc[:, 2], linewidth=2, color='steelblue')
            ax.set_title(metric, fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)

        except FileNotFoundError:
            ax.text(0.5, 0.5, f'{filename}\nNOT FOUND',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=10, color='red')

    plt.tight_layout()
    plt.savefig(f'{model}_metrics_CIFAR100.png', dpi=300, bbox_inches='tight')
    print(f'Saved: {model}_metrics.png')
    plt.close()




print('Done! Check your directory for PNG files.')