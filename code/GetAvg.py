import glob
import pandas as pd
import os
from pathlib import Path

output_path = Path('../output/')
experiment_names = ['.'.join(ex_name.split('.')[:-2]) for ex_name in glob.glob(str(output_path / '*'))]
experiment_names = set(experiment_names)

for experiment_name in experiment_names:
    for output_name in ['summary', 'test_summary']:
        filenames = glob.glob(experiment_name + '*/result_outputs/' + output_name + '.csv')

        dfs = []
        for filename in filenames:
            dfs.append(pd.read_csv(filename))
        df_concat = pd.concat(dfs, axis=1)
        means_df = df_concat.stack().groupby(level=[0,1]).mean().unstack()
        means_df.columns = [str(col) + '_mean' for col in means_df.columns]
        sem_df = df_concat.stack().groupby(level=[0,1]).sem().unstack()
        sem_df.columns = [str(col) + '_sem' for col in sem_df.columns]
        final_df = pd.concat((means_df, sem_df), axis=1)
        if not os.path.exists(Path(experiment_name + '.avg')):
            os.makedirs(Path(experiment_name + '.avg'))

        output_file = Path(experiment_name + '.avg') / (output_name + '.csv')
        final_df.index += 1
        final_df.to_csv(output_file, index_label='epoch', float_format='%.4f')
