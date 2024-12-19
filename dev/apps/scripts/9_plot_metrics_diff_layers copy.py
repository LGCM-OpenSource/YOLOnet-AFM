import pandas as pd 
from utils import DataChart, EvalModel
import os 
import numpy as np 
chart = DataChart()


df_metrics_list = []


models_list = [
                'Unet_2_layers',
                'Unet_3_layers',
                'Unet_4_layers',
                'Unet_5_layers',
                
               ]

df_results = pd.read_csv('/home/arthur/lgcm/projects/Segmentation_union_projects/all_data_predict.csv')


title_chart = 'Different Unet Architecutes comparison'

for model in models_list:
    df_path = model
    y_true = df_results['y_true'].values
    y_pred = df_results[model].values > 0.5
    y_pred.astype(np.uint8)
    # df_metrics_general_validation.append(df)

    eval = EvalModel('unet',y_true, y_pred) 
    scores = eval.get_metrics()
    metric_df = eval.metrics_to_df(scores)
    metric_df = metric_df.replace({'Model':'unet'},model)
    
        
    df_metrics_list.append(metric_df)
        
final_df = pd.concat(df_metrics_list, axis=0)


final_df_melt_general = pd.melt(final_df, id_vars=['Process Date','Model'], value_vars=['Precision', 'Recall', 'F1', 'Dice'], var_name = 'metrics', value_name = 'scores')    
# final_df_melt = pd.melt(final_df, id_vars=['Process Date','Model'], value_vars=['Precision', 'Recall', 'F1', 'Dice'], var_name = 'metrics', value_name = 'scores')
    
fig = chart.box_plot(final_df_melt_general, x='Model', y='scores', color = 'metrics',
                     width=800, height = 500,
                     title='', 
                     x_label=False, y_label=False)

fig2 = chart.box_plot(final_df_melt_general, x='Model', y='scores', color = 'metrics',
                      width=800, height = 600,
                      title=title_chart)

# fig3 = chart.violin_plot(final_df, x='Model', y='scores', color = 'metrics',
#                       width=800, height = 600,
#                       title=title_chart)

fig.write_image(f'{title_chart}.svg')
fig2.write_image(f'{title_chart}.png')






