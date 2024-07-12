import os
import numpy as np
from utils import UnetProcess, EvalModel
from tqdm import tqdm
from glob import glob
import matplotlib.pyplot as plt 
import pandas as pd 

def dice(y_pred, y, k = True):
        intersection = np.sum(y_pred[y==k]) * 2.0
        dice = intersection / (np.sum(y_pred) + np.sum(y))
        return round(dice,2)

def precision(y_pred, y):
        negative = False
        positive = True
        tp = np.sum(np.logical_and(y_pred == positive, y == positive))
        tn = np.sum(np.logical_and(y_pred == negative, y == negative))
        fp = np.sum(np.logical_and(y_pred == positive, y == negative))
        fn = np.sum(np.logical_and(y_pred == negative, y == positive))
        precision_score = tp / (tp + fp)
        return round(precision_score,2)
        
def recall(y_pred, y):
        negative = False
        positive = True
        tp = np.sum(np.logical_and(y_pred == positive, y == positive))
        tn = np.sum(np.logical_and(y_pred == negative, y == negative))
        fp = np.sum(np.logical_and(y_pred == positive, y == negative))
        fn = np.sum(np.logical_and(y_pred == negative, y == positive))
        recall_score =  tp / (tp + fn)
        return round(recall_score,2)

def f1(y_pred, y):
        negative = False
        positive = True
        tp = np.sum(np.logical_and(y_pred == positive, y == positive))
        tn = np.sum(np.logical_and(y_pred == negative, y == negative))
        fp = np.sum(np.logical_and(y_pred == positive, y == negative))
        fn = np.sum(np.logical_and(y_pred == negative, y == positive))
        f1_score =  2*tp / (2*tp + fp + fn)
        return round(f1_score,2)






model_list = ['unet_afm_6_channels_final_model.h5' , 'UNet_AFM_6_channels_5_layers_final_model.h5']


for model in model_list:
        # if model != 'unet_afm_6_channels_150_images.h5':
        #         continue
        print(f'get {model} validation... ')
        model_name = model
        
        if model == 'unet_afm_6_channels_226_images.h5':
                model_name = 'UNet_AFM_6_channels_cv_100_epoch2.h5'
                
                
        preprocess_image = f'data{os.sep}intermediate{os.sep}pre_processing_optico_and_afm{os.sep}image{os.sep}'
        mask = f'data{os.sep}intermediate{os.sep}pre_processing_optico_and_afm{os.sep}mask{os.sep}'
        opt_image = f'data{os.sep}input{os.sep}optical_images_resized{os.sep}'
        # save_path = f'data{os.sep}output{os.sep}{unet_afm_fold}{os.sep}predicts{os.sep}'
        predict_path = f'data{os.sep}input{os.sep}Usefull_data{os.sep}'



        
                
        dire = os.listdir(preprocess_image)

        opt_image_path = [opt_image + file.replace('_channels_added.npy', '_optico_crop_resized.png') for file in dire]
        preprocess_image_path = [preprocess_image+file for file in dire]
        usefull_path = [predict_path+file.replace('_channels_added.npy', '_UsefullData.tsv') for file in dire]
        mask_path = [mask+file for file in dire]
        # save_path = [save_path+file.replace('_channels_added.npy', '_unet.png') for file in dire]

        df_list = []

        for i in tqdm(range(len(opt_image_path)), colour='#0000FF'):
                process_data = opt_image_path[i].split(f'{os.sep}')[-1]
                unetTrat =   UnetProcess(opt_image_path[i], preprocess_image_path[i], usefull_path[i], mask_path[i], model_path=f'models{os.sep}{model_name}') 
                
                # usefull_path_unet = usefull_path[i].replace(f'data{os.sep}input{os.sep}Usefull_data{os.sep}',f'data{os.sep}output{os.sep}{unet_afm_fold}{os.sep}predict_sheets{os.sep}')
                y,y_pred, _t = unetTrat.unet_predict()
                y_flatten = y.flatten()
                y_pred_flatten = y_pred.flatten()

                
                eval = EvalModel('unet',process_data, y_flatten, y_pred_flatten) 
                scores = eval.get_metrics()
                metric_df = eval.metrics_to_df(scores)
                df_list.append(metric_df)
        final_metric_df = pd.concat(df_list, axis=0)
        final_metrics_melt = pd.melt(final_metric_df, id_vars=['Process Date','Model'], value_vars=['Precision', 'Recall', 'F1', 'Dice'], var_name = 'metrics', value_name = 'scores')
        final_metrics_melt.to_csv(f'test_metrics_{model_name.split(".")[0]}.csv')        
        
        
        
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