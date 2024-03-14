import os 
import sys
sys.path.append(f'dev{os.sep}scripts')
from dataframe_treatment import DataFrameTrat
from models import EvalModel
import pandas as pd
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-op', '--options', type=int)

args = parser.parse_args()
option = args.options
# option = 0

predicts_dict = {'vUnet_AFM': f'data{os.sep}output{os.sep}vunet_AFM_predictions{os.sep}predict_sheets{os.sep}',
                 'Unet_AFM':  f'data{os.sep}output{os.sep}unet_AFM_predictions{os.sep}predict_sheets{os.sep}',
                 'Only_AFM': f'data{os.sep}output{os.sep}only_afm_predictions{os.sep}predict_sheets{os.sep}'
                 }
save_dict = {
            'vUnet_AFM':f'data{os.sep}output{os.sep}vunet_AFM_predictions{os.sep}vUnet_AFM_metrics.csv', 
            'Unet_AFM': f'data{os.sep}output{os.sep}unet_AFM_predictions{os.sep}Unet_AFM_metrics.csv',
            'Only_AFM': f'data{os.sep}output{os.sep}only_afm_predictions{os.sep}predict_sheets{os.sep}'
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
    prediction_path = [predicts_dict['Unet_AFM']]
    save_path = [save_dict['Unet_AFM']]
    
elif option == 3:
    prediction_path = [predicts_dict['Only_AFM']]
    save_path = [save_dict['Only_AFM']]


for path, save in zip(prediction_path, save_path):
    prediction_files = os.listdir(path)
    df_list = []
    for file in tqdm(prediction_files):
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

            eval = EvalModel(type_model, y_true, y_pred)

            scores = eval.get_metrics()
            metric_df = eval.metrics_to_df(process_date, scores)
            df_list.append(metric_df)
    
    final_df = pd.concat(df_list, axis=0)
    eval.save_metrics(final_df, save_path=save)
    print(f'Metrics Saved in "{save}"')

sys.stdout.close()