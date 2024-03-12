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
sys.path.append(f'dev{os.sep}scripts')
from colors_to_pcr import HEXA_COLORS
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


while True:
    print(
        '''
        Select a number according to which model you want to evaluate:\n
        1 - vUnet_AFM
        2 - Unet_AFM
        3 - Pixel_AFM
        '''
        )
    option  = input('Enter the number of the desired option:\n')
    
    if option.isdigit() and int(option) in [1, 2, 3]:
        option = int(option)
        break
    else: 
        print("Enter a valid option:\n")
if option == 1:
    img_path = f'data{os.sep}output{os.sep}vunet_AFM_predictions{os.sep}'
    result_path = f'data{os.sep}output{os.sep}vunet_AFM_predictions{os.sep}predict_sheets{os.sep}'
    final_metrics_results_path = f"data{os.sep}output{os.sep}vunet_AFM_predictions{os.sep}metric_results{os.sep}"
    
elif option == 2:
    img_path = f'data{os.sep}output{os.sep}unet_AFM_predictions{os.sep}'
    result_path = f'data{os.sep}output{os.sep}unet_AFM_predictions{os.sep}predict_sheets{os.sep}'
    final_metrics_results_path = f"data{os.sep}output{os.sep}unet_AFM_predictions{os.sep}metric_results{os.sep}"
    
elif option == 3:
    img_path = f'data{os.sep}output{os.sep}only_afm_predictions{os.sep}'
    result_path = f'data{os.sep}output{os.sep}only_afm_predictions{os.sep}predict_sheets{os.sep}'
    final_metrics_results_path = f"data{os.sep}output{os.sep}only_afm_predictions{os.sep}metric_results{os.sep}"
    


results_files = os.listdir(img_path)


for img in results_files:
    if os.path.isfile(img_path+img):
        process_date = img.split('_')[0]
        model_name = img.split('_')[1]
        

        if len(img.split('_'))>2:
            model_name = img.split('_')[1:3]
            model_name = '_'.join(model_name)
        
        df_name = img.replace(f'_{model_name}', '_UsefullData.tsv')
        df_path = f'{result_path}{df_name}'
        
        df_treat = DataFrameTrat(df_path)
        df_predict = df_treat.df
        df_predict = df_treat.clean_target(df_predict)
        
        #Remove .png extension from model_name
        model_name = model_name.split('.')[0]
        column = model_name
        if model_name == 'unet':
            column = model_name+'_prediction'
        else:
            print('aqyu')
            
            
        y = df_predict['Generic Segmentation'].copy()
        y_pred = df_predict[column].copy()
        
        unet, voted = select_model(model_name)
        
        """ Calculating metrics values """
        SCORE=[]
        acc_value, f1_value, jac_value, recall_value, precision_value = get_scores(y, y_pred)
        # unet_acc_value, unet_f1_value, unet_jac_value, unet_recall_value, unet_precision_value = get_scores(y, y_unet)
        
        SCORE.append([process_date, unet, voted, model_name, jac_value, recall_value, precision_value, f1_value])
        # if y_unet.equals(y_pred):
        #     SCORE=[]
        #     SCORE.append([process_date, unet, voted, model_name, jac_value, recall_value, precision_value, f1_value])
        
        
        
        df_metrics, data_to_chart = metrics_df(SCORE, unet=unet)
        # df_metrics.to_csv(f'data{os.sep}output{os.sep}final_prediction_sheets{os.sep}{process_date}_{model_name}.tsv', index=False)
        

        img = open_image(f'{img_path}{img}')
        # if column != 'unet_predict':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        fig = make_subplots(1, 2)
        # We use go.Image because subplots require traces, whereas px functions return a figure
        fig.add_trace(go.Image(z=img), 1, 1)
        fig.add_trace(go.Bar(x = data_to_chart["Metrics"],
                                y = data_to_chart['Scores'],
                            marker_color = [HEXA_COLORS['Blue'], HEXA_COLORS['Red'], HEXA_COLORS['Green']],),1,2)
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
        fig.write_image(f"{final_metrics_results_path}{process_date}_{model_name}.svg") 













# for file in results_files:
#     process_date = file.split('_')[0]
#     img_name = file.replace('_UsefullData.tsv', '_unet.png')
#     df_path = f'{result_path}{file}'
#     df_result = pd.read_csv(df_path, sep=',', index_col=0)
    
#     #Separate prediction and real segmentation
#     y = df_result['Generic Segmentation']
    
    
#     for pred in df_result.columns:
#         SCORE = []
#         if pred != 'Generic Segmentation':
#             model = pred.replace('_predict', '')
#             y_pred = df_result[pred]
            
            
#             """ Calculating metrics values """
#             acc_value = accuracy_score(y, y_pred)
#             f1_value = f1_score(y, y_pred, labels=np.unique(y_pred), average="binary",zero_division=0)
#             jac_value = jaccard_score(y, y_pred, labels=np.unique(y_pred), average="binary",zero_division=0)
#             recall_value = recall_score(y, y_pred, labels=np.unique(y_pred), average="binary",zero_division=0)
#             precision_value = precision_score(y, y_pred, labels=np.unique(y_pred), average="binary",zero_division=0)
#             SCORE.append([model,process_date, acc_value, f1_value, jac_value, recall_value, precision_value])
#             df = pd.DataFrame(SCORE, columns=["Model", "Process Date", "Accuracy", "F1", "Jaccard", "Recall", "Precision"])
#             tes = pd.melt(df, id_vars='Model', var_name = 'Metrics', value_name = 'Scores')
#             tes.drop(0, inplace=True)
            
#             img = cv2.imread(f'{img_path}{img_name}', cv2.IMREAD_COLOR)
#             fig = make_subplots(1, 2)
#             # We use go.Image because subplots require traces, whereas px functions return a figure
#             fig.add_trace(go.Image(z=img), 1, 1)
#             fig.add_trace(go.Bar(x = tes["Metrics"],
#                                  y = tes['Scores'],
#                                 marker_color = '#1f77b4',),1,2)
#             fig.update_layout({
           
#             'title': f'{model}',
#             })
#             # fig.show()
#             fig.write_image(f"data{os.sep}output{os.sep}metric_results{os.sep}{process_date}_{model}.png")
# #            