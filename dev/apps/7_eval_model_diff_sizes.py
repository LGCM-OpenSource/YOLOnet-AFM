import os 
import sys
from utils import DataFrameTrat, EvalModel
import pandas as pd
from tqdm import tqdm
import argparse
import traceback

parser = argparse.ArgumentParser()
parser.add_argument('-op', '--options', type=int)

args = parser.parse_args()
option = 2


predicts_dict = {'vUnet_AFM': f'data{os.sep}output{os.sep}vunet_AFM_predictions{os.sep}predict_sheets{os.sep}',
                 'Unet_AFM':  [
                               f'data{os.sep}output{os.sep}unet_AFM_predictions_50_images{os.sep}predict_sheets{os.sep}',
                               f'data{os.sep}output{os.sep}unet_AFM_predictions_100_images{os.sep}predict_sheets{os.sep}',
                               f'data{os.sep}output{os.sep}unet_AFM_predictions_150_images{os.sep}predict_sheets{os.sep}',
                               f'data{os.sep}output{os.sep}unet_AFM_predictions_200_images{os.sep}predict_sheets{os.sep}',
                               ],
                 'Only_AFM': f'data{os.sep}output{os.sep}only_afm_predictions{os.sep}predict_sheets{os.sep}'
                 }
save_dict = {
            'vUnet_AFM':f'data{os.sep}output{os.sep}vunet_AFM_predictions{os.sep}vUnet_AFM_metrics.csv', 
            'Unet_AFM': [
                        f'unet_AFM_metrics_50_images.csv',
                        f'unet_AFM_metrics_100_images.csv',
                        f'unet_AFM_metrics_150_images.csv',
                        f'unet_AFM_metrics_200_images.csv',
                        ],
            
            'Only_AFM': f'data{os.sep}output{os.sep}only_afm_predictions{os.sep}AFM_only_metrics.csv'
            }


if option == 0:
    prediction_path = [
                      predicts_dict['vUnet_AFM'],
                      predicts_dict['Unet_AFM'],
                      predicts_dict['Only_AFM']
                       ]
    
    save_path = [
                save_dict['vUnet_AFM'],
                save_dict['Unet_AFM'],
                save_dict['Only_AFM']
                ]

elif option == 1:
    prediction_path = [predicts_dict['vUnet_AFM']]
    save_path = [save_dict['vUnet_AFM']]
    
elif option == 2:
    prediction_path = predicts_dict['Unet_AFM']
    save_path = save_dict['Unet_AFM']
    
elif option == 3:
    prediction_path = [predicts_dict['Only_AFM']]
    save_path = [save_dict['Only_AFM']]


for path, save_p in zip(prediction_path, save_path): 
    prediction_files = os.listdir(path)
    df_list = []
    for file in tqdm(prediction_files, colour='#0000FF'):
        try:
            if os.path.isfile(path+file):
                
                process_date = file.replace('_UsefullData.tsv', '')
                df_predict = DataFrameTrat(os.path.join(path, file))

                df = df_predict.df 
                df = df_predict.clean_target(df)

                model_name = df.columns[-1].split('_')[0:2]
                type_model = '_'.join(model_name)

                # get y_true e y_pred
                y_true = df['Generic Segmentation']
                y_pred = df[df.columns[-1]]

                eval = EvalModel(type_model,process_date, y_true, y_pred)

                scores = eval.get_metrics()
                metric_df = eval.metrics_to_df(scores)
                df_list.append(metric_df)
        except Exception: 
            print(traceback.format_exc())
            
    final_df = pd.concat(df_list, axis=0)
    final_df.to_csv(save_p)
    print(f'Metrics Saved in "{save_p}"')
  
sys.stdout.close()