import pandas as pd 
from utils import DataChart
import os 
chart = DataChart()


df_metrics_list = []
for fold in range(1,11):
    df_path = f'models{os.sep}fold_{fold}_cv_6_channels_training.log'
    df = pd.read_csv(df_path, index_col=0).reset_index()
    if fold != 1:
        df = df.rename(columns={f'precision_{fold - 1}': 'precision',  f'recall_{fold - 1}': 'recall',  f'val_precision_{fold - 1}':'val_precision',  f'val_recall_{fold - 1}':'val_recall'})

    df = df.iloc[[-1]]


    df_melt= pd.melt(df, id_vars=['epoch'], value_vars=['val_precision', 'val_recall', 'val_dice_coef'], var_name = 'metrics', value_name = 'scores')    
    df_metrics_list.append(df_melt)

final_df_metrics = pd.concat(df_metrics_list, axis=0)
final_df_metrics['epoch'] = 'cross validation'
fig = chart.box_plot(final_df_metrics, x='epoch', y='scores', color = 'metrics', 
                     title=f'Unet_AFM_6_channels_validation_metrics_cross_validation')


fig.write_image(f'Unet_AFM_6_channels_validation_metrics_cross_validation.png')

    