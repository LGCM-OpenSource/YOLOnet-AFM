from models import EvalModel
import os 
import sys
sys.path.append(f'dev{os.sep}scripts')
from dataframe_treatment import DataFrameTrat
import pandas as pd
from tqdm import tqdm

prediction_path = 'data/output/predict_sheets/'
save_path = 'metrics.csv'

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







