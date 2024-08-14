import os
import numpy as np
from utils import UnetProcess, EvalModel
from tqdm import tqdm
from glob import glob
import matplotlib.pyplot as plt 
import pandas as pd 



# images_to_compare = ['afm_optico_ok', 'artifacts']


# for folder in images_to_compare:


afm_info_path = f'data_complete/intermediate/pre_processing_optico_and_afm/image/'
masks_path =  f'data_complete/intermediate/pre_processing_optico_and_afm/mask/'

model_name = afm_info_path.split('/')[1]
dire = os.listdir(afm_info_path)

process_data = [file.split('_')[0] for file in dire]
preprocess_image = [np.load(afm_info_path+file).astype(np.uint8) for file in dire]
mask = [np.load(masks_path+file).astype(np.uint8) for file in dire]
# save_path = [save_path+file.replace('_channels_added.npy', '_unet.png') for file in dire]

df_list = []

for i in tqdm(range(len(preprocess_image)), colour='#0000FF'):
        # usefull_path_unet = usefull_path[i].replace(f'data{os.sep}input{os.sep}Usefull_data{os.sep}',f'data{os.sep}output{os.sep}{unet_afm_fold}{os.sep}predict_sheets{os.sep}')
        y_AFM = preprocess_image[i]
        y_AFM_flatten = y_AFM.flatten()
        y_AFM_flatten[y_AFM_flatten>0] = 1

        y = mask[i]
        y_flatten = y.flatten()

        
        eval = EvalModel('unet',process_data[i], y_flatten, y_AFM_flatten) 
        scores = eval.get_metrics()
        metric_df = eval.metrics_to_df(scores)
        df_list.append(metric_df)
final_metric_df = pd.concat(df_list, axis=0)
final_metrics_melt = pd.melt(final_metric_df, id_vars=['Process Date','Model'], value_vars=['Precision', 'Recall', 'F1', 'Dice'], var_name = 'metrics', value_name = 'scores')
final_metrics_melt.to_csv(f'AFM_mask_validation_corrected_{model_name.split(".")[0]}.csv')        


        
## manually metrics calc
        # f1_score = f1(y_pred_flatten, y_flatten)
        # dice_score = dice(y_pred_flatten, y_flatten)
        # precision_score = precision(y_pred_flatten, y_flatten)
        # recall_score = recall(y_pred_flatten, y_flatten)
        
        # fig, axs = plt.subplots(1, 2)
        # axs[0].imshow(y, cmap='gray')
        # axs[0].axis('off')  # Desativar os eixos
        # axs[0].set_title(f'Manually calc:\nPrecision: {precision_score}\nRecall: {recall_score}\nF1: 0.86\nDice: {dice_score}', fontsize=10, loc='left')

        # axs[1].imshow(y_pred, cmap='gray')
        # axs[1].axis('off')  # Desativar os eixos
        # axs[1].set_title(f'{metric_df}', fontsize=10, loc='left')
        # plt.show()
        # plt.close()
        #         # Definir os cabeçalhos das colunas
        # header = "Model  Precision    Recall        F1   Dice"

        # separator = "-" * len(header)

        # data_row = f"{'unet':<6}  {precision_score:<10}  {recall_score:<10}  {f1_score:<10}  {dice_score:<10}\n"

        # # print table
        # print('Manually metrics calc')
        # print(header)
        # print(separator)
        # print(data_row)
        
        
        
        # # print(F"IMAGEM: {process_data}\nPRECISION:  {precision_score}\nRECALL: {recall_score}\nF1: {f1_score}\nDICE: {dice_score}\n")
        # print('Lib metrics calc')
        # print(metric_df)