import glob
import pandas as pd
import os
from pathlib import Path

output_path = Path('../output/')
experiment_names = ['.'.join(ex_name.split('.')[:-2]) for ex_name in glob.glob(str(output_path / '*'))]
for experiment_name in experiment_names:
    if ('base.' not in experiment_name) or ('exp.' not in experiment_names):
        experiment_names.remove(experiment_name)
experiment_names = set(experiment_names)

for experiment_name in experiment_names:
    for output_name in ['summary']:
        filenames = glob.glob(experiment_name + '*/result_outputs/' + output_name + '.csv')

        dfs = []
        for filename in filenames:
            file_dfs = pd.read_csv(filename)
            file_dfs['cum_time'] = file_dfs.epoch_time.cumsum()
            dfs.append(file_dfs)
        df_concat = pd.concat(dfs, axis=1)
        means_df = df_concat.stack().groupby(level=[0,1]).mean().unstack()
        means_df.columns = [str(col) + '_mean' for col in means_df.columns]
        sem_df = df_concat.stack().groupby(level=[0,1]).sem().unstack()
        sem_df.columns = [str(col) + '_sem' for col in sem_df.columns]
        final_df = pd.concat((means_df, sem_df), axis=1)
        if not os.path.exists(Path(experiment_name + '.avg')):
            os.makedirs(Path(experiment_name + '.avg'))

        output_file = Path(experiment_name + '.avg') / (output_name + '.cumu.time.csv')
        final_df.index += 1
        final_df.to_csv(output_file, index_label='epoch', float_format='%.4f',
                        columns=['cum_time_mean', 'cum_time_sem'])
