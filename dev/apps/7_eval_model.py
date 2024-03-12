import os 
import sys
sys.path.append(f'dev{os.sep}scripts')
from dataframe_treatment import DataFrameTrat
from models import EvalModel
import pandas as pd
from tqdm import tqdm


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
    prediction_path = f'data{os.sep}output{os.sep}vunet_AFM_predictions{os.sep}predict_sheets{os.sep}'
    save_path = 'vUnet_AFM_metrics.csv'
    
elif option == 2:
    prediction_path = f'data{os.sep}output{os.sep}unet_AFM_predictions{os.sep}predict_sheets{os.sep}'
    save_path = 'unet_AFM_metrics.csv'
    
elif option == 3:
    img_path = f'data{os.sep}output{os.sep}only_afm_predictions{os.sep}'
    prediction_path = f'data{os.sep}output{os.sep}only_afm_predictions{os.sep}predict_sheets{os.sep}'
    save_path = 'pixel_AFM_metrics.csv'
    

prediction_files = os.listdir(prediction_path)

df_list = []
for file in tqdm(prediction_files):
    process_date = file.replace('_UsefullData.tsv', '')
    df_predict = DataFrameTrat(os.path.join(prediction_path, file))

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
eval.save_metrics(final_df, save_path=save_path)







