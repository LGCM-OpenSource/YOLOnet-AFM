import pandas as pd 
from utils import DataChart
import os 
chart = DataChart()


df_metrics_list = []
for fold in range(1,11):
    df_path = f'Unet_AFM_{fold}fold_metrics2.csv'
    df = pd.read_csv(df_path, index_col=0)
    df = df.replace({'Model':'unet_prediction'},f'unet_{fold}fold')
    df_metrics_list.append(df)
final_df = pd.concat(df_metrics_list, axis=0)
    
final_df_melt = pd.melt(final_df, id_vars=['Process Date','Model'], value_vars=['Precision', 'Recall', 'F1', 'Dice'], var_name = 'metrics', value_name = 'scores')
    
fig = chart.box_plot(final_df_melt, x='Model', y='scores', color = 'metrics', title='')
# fig.show()
fig.write_image(f'unet_AFM_different_folds2.png')