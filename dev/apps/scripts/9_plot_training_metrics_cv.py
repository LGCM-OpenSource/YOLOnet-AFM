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




    df_melt_loss = pd.melt(df, id_vars=['epoch'], value_vars=['loss', 'val_loss'], var_name = 'metrics', value_name = 'scores')    
    df_melt_dice = pd.melt(df, id_vars=['epoch'], value_vars=['dice_coef', 'val_dice_coef',], var_name = 'metrics', value_name = 'scores')    
    df_melt_precision = pd.melt(df, id_vars=['epoch'], value_vars=['precision', 'val_precision'], var_name = 'metrics', value_name = 'scores')    
    df_melt_recall = pd.melt(df, id_vars=['epoch'], value_vars=['recall','val_recall'], var_name = 'metrics', value_name = 'scores')    

    list_df_validation_metrics = [('loss_X_val_loss',df_melt_loss), ('dice_coef_X_val_dice_coef',df_melt_dice), ('precision_X_val_precision',df_melt_precision), ('recall_X_val_recall',df_melt_recall)]
    for title, dfs in list_df_validation_metrics:
        fig = chart.line_plot(dfs, x='epoch', y='scores', color = 'metrics', title=f'Unet_AFM_6_channels_{fold}_training_metrics_cross_validation: {title}')


        fig.write_image(f'Unet_AFM_6_channels_training_metrics_cross_validation.png')
    
    