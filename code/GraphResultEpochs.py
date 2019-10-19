import glob
import pandas as pd
import os
from pathlib import Path
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams.update({'font.size': 13})
import matplotlib.pyplot as plt

graphs_folder = Path('../graphs/')
fig, ax = plt.subplots()
# Set the labels
ax.xaxis.set_label_text('Epoch')
ax.yaxis.set_label_text('Validation accuracy')
ax.set_ylim(0.82, 0.9)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:3]

output_path = Path('../output/')
experiment_names = glob.glob(str(output_path / '*.avg'))

for experiment_name in experiment_names:
    dfs = pd.read_csv(experiment_name + '/summary.csv')
    line = ax.plot(dfs['epoch'].values, dfs['val_acc_mean'].values,
                   linewidth=1)[0]
    label = ''
    if 'dilated_convolution' in experiment_name:
        line.set_color(colors[0])
        label += 'Dilation'
    elif 'max_pooling' in experiment_name:
        line.set_color(colors[1])
        label += 'Max-pooling'
    elif 'strided_convolution' in experiment_name:
        line.set_color(colors[2])
        label += 'Striding'

    if 'base' in experiment_name:
        line.set_linestyle('--')
        label += ' baseline'

    line.set_label(label)

ax.legend()
fig.savefig(str(graphs_folder / 'valEpochs.eps'), bbox_inches='tight')
fig.show()
