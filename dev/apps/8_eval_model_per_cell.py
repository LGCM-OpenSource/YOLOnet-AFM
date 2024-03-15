import os
import pandas as pd
import numpy as np
# import plotly packages 
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import cv2
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score, roc_curve
import sys
import argparse
import traceback
from tqdm import tqdm

sys.path.append(f'dev{os.sep}scripts')
from colors_to_pcr import COLOR_METRICS_STYLE
from dataframe_treatment import DataFrameTrat


def open_image(path):
    img = cv2.imread(path)
    return img

def get_scores(y, y_pred):
        """ Calculating metrics values """
        acc_value = accuracy_score(y, y_pred)
        f1_value = f1_score(y, y_pred, labels=np.unique(y_pred), average="binary",zero_division=0)
        jac_value = jaccard_score(y, y_pred, labels=np.unique(y_pred), average="binary",zero_division=0)
        recall_value = recall_score(y, y_pred, labels=np.unique(y_pred), average="binary",zero_division=0)
        precision_value = precision_score(y, y_pred, labels=np.unique(y_pred), average="binary",zero_division=0)
        
        return acc_value, f1_value, jac_value, recall_value, precision_value

def metrics_df(scores, unet=False):
    # if unet:
    df = pd.DataFrame(scores, columns=["Process Date", "Unet", "Voted", "Model" , "Jaccard", "Recall", "Precision", "F1"])
    # else:
    #     df = pd.DataFrame(scores, columns=["Process Date", "Unet", "Voted", "Model" , "Jaccard", "Recall", "Precision", "F1", "unet_Jaccard", "unet_Recall", "unet_Precision", "unet_F1"])
    tes = pd.melt(df[["Model" ,"Precision", "Recall",  "F1"]].copy(), id_vars='Model', var_name = 'Metrics', value_name = 'Scores')
    return df, tes  

def select_model(model_name):
    if model_name != 'unet':
        return False, True 
    return True, False

parser = argparse.ArgumentParser()
parser.add_argument('-op', '--options', type=int)

args = parser.parse_args()
option = args.options
dict_dirs = {'img_path':
                            {
                            'vUnet_AFM':f'data{os.sep}output{os.sep}vunet_AFM_predictions{os.sep}predicts{os.sep}',
                            'Unet_AFM': f'data{os.sep}output{os.sep}unet_AFM_predictions{os.sep}predicts{os.sep}',
                            'Only_AFM': f'data{os.sep}output{os.sep}only_afm_predictions{os.sep}predicts{os.sep}'
                            }
                        ,
             'result_path': {
                            'vUnet_AFM':f'data{os.sep}output{os.sep}vunet_AFM_predictions{os.sep}predict_sheets{os.sep}',
                            'Unet_AFM': f'data{os.sep}output{os.sep}unet_AFM_predictions{os.sep}predict_sheets{os.sep}',
                            'Only_AFM': f'data{os.sep}output{os.sep}only_afm_predictions{os.sep}predict_sheets{os.sep}'
                            },
             'final_metrics_results_path':{
                                            'vUnet_AFM':f"data{os.sep}output{os.sep}vunet_AFM_predictions{os.sep}metrics_per_cell{os.sep}",
                                            'Unet_AFM': f"data{os.sep}output{os.sep}unet_AFM_predictions{os.sep}metrics_per_cell{os.sep}", 
                                            'Only_AFM': f"data{os.sep}output{os.sep}only_afm_predictions{os.sep}metrics_per_cell{os.sep}"

                                           }
             
             }

if option ==0:
    img_path = [
                    dict_dirs['img_path']['vUnet_AFM'],
                    dict_dirs['img_path']['Unet_AFM'],
                    dict_dirs['img_path']['Only_AFM']        
                ]
    result_path = [
                    dict_dirs['result_path']['vUnet_AFM'],
                    dict_dirs['result_path']['Unet_AFM'],
                    dict_dirs['result_path']['Only_AFM'] 
                    ]
    final_metrics_results_path = [
                                    dict_dirs['final_metrics_results_path']['vUnet_AFM'],
                                    dict_dirs['final_metrics_results_path']['Unet_AFM'],
                                    dict_dirs['final_metrics_results_path']['Only_AFM'] 
                                    ]
elif option == 1:
    img_path = [dict_dirs['img_path']['vUnet_AFM']]
    result_path = [dict_dirs['result_path']['vUnet_AFM']]
    final_metrics_results_path = [dict_dirs['final_metrics_results_path']['vUnet_AFM']]
    
elif option == 2:
    img_path = [dict_dirs['img_path']['Unet_AFM']]
    result_path = [dict_dirs['result_path']['Unet_AFM']]
    final_metrics_results_path = [dict_dirs['final_metrics_results_path']['Unet_AFM']]
    
elif option == 3:
    img_path = [dict_dirs['img_path']['Only_AFM']]
    result_path = [dict_dirs['result_path']['Only_AFM']]
    final_metrics_results_path = [dict_dirs['final_metrics_results_path']['Only_AFM']]
 
 
for  img_p, result_p, final_p in tqdm(zip(img_path, result_path, final_metrics_results_path), colour='green',):
    results_files = os.listdir(img_p)

    for img in tqdm(results_files, colour='#0000FF'):
        try:
            if os.path.isfile(img_p+img):
                process_date = img.split('_')[0]
                model_name = img.split('_')[1]
                

                if len(img.split('_'))>2:
                    model_name = img.split('_')[1:3]
                    model_name = '_'.join(model_name)
                
                df_name = img.replace(f'_{model_name}', '_UsefullData.tsv')
                df_path = f'{result_p}{df_name}'
                
                df_treat = DataFrameTrat(df_path)
                df_predict = df_treat.df
                df_predict = df_treat.clean_target(df_predict)
                
                #Remove .png extension from model_name
                model_name = model_name.split('.')[0]
                column = model_name
                if model_name == 'unet':
                    column = model_name+'_prediction'
                
                y = df_predict['Generic Segmentation'].copy()
                y_pred = df_predict[column].copy()
                
                unet, voted = select_model(model_name)
                
                """ Calculating metrics values """
                SCORE=[]
                acc_value, f1_value, jac_value, recall_value, precision_value = get_scores(y, y_pred)
                
                SCORE.append([process_date, unet, voted, model_name, jac_value, recall_value, precision_value, f1_value])
                
                df_metrics, data_to_chart = metrics_df(SCORE, unet=unet)
                # df_metrics.to_csv(f'data{os.sep}output{os.sep}final_prediction_sheets{os.sep}{process_date}_{model_name}.tsv', index=False)
                

                img = open_image(f'{img_p}{img}')
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                fig = make_subplots(1, 2)
                # We use go.Image because subplots require traces, whereas px functions return a figure
                fig.add_trace(go.Image(z=img), 1, 1)
                fig.add_trace(go.Bar(x = data_to_chart["Metrics"],
                                        y = data_to_chart['Scores'],
                                    marker_color = [COLOR_METRICS_STYLE['Precision'], COLOR_METRICS_STYLE['Recall'], COLOR_METRICS_STYLE['F1']]),1,2)
                fig.update_yaxes(range = [0,1], row=1, col=2)
                fig.update_xaxes(categoryorder='array', categoryarray= ['Precision','Recall','F1'])
                fig.update_layout({
                                            
                                            'title': f'Metrics from {model_name}',
                                            },
                                            plot_bgcolor='white',
                                            font_family="Arial",
                                            font = dict(size=10),
                                            title_font_family="Arial",
                                            )
                fig.update_yaxes(
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor='black',
                gridcolor='lightgrey'
                )

                fig.update_xaxes(
                    mirror=True,
                    ticks='outside',
                    showline=True,
                    linecolor='black',
                    gridcolor='lightgrey'
                )
                # fig.show()
                fig.write_image(f"{final_p}{process_date}_{model_name}.svg") 
        except Exception:
            print(traceback.format_exc())

    print(f'Metrics Saved in "{final_p}"')

sys.stdout.close()