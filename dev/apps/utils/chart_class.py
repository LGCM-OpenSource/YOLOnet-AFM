import pandas as pd  
import os 
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import plotly.graph_objects as go


class DataChart:
    def __init__(self, data_path=False, sep=False):
        self.dada_path = data_path
        self.sep = sep
    
    @property
    def df(self):
        dataframe = pd.read_csv(self.dada_path)
        if self.sep:
            dataframe = pd.read_csv(self.dada_path, sep=self.sep)
            return dataframe
        if 'Unnamed: 0' in dataframe.columns:
            return pd.read_csv(self.dada_path, index_col=0)
        else:
            return dataframe
            
    def apply_melt(self, df, id_vars = ['Model', 'model_len', 'Process Date'], value_vars=['Jaccard',	'Recall',	'Precision',	'F1', 'unet_F1', 'unet_Jaccard', 'unet_Precision', 'unet_Recall'] , var_name= 'metrics' , value_name='scores'):
         return pd.melt(df, id_vars=id_vars, value_vars=value_vars , var_name= var_name , value_name=value_name)   
    
    def ordenate_metrics(self, df, column, metrics_order = ['Precision'  ,    'Recall'  ,  'F1'  ,  'Jaccard', 'unet_Precision'  ,    'unet_Recall'  ,  'unet_F1'  ,  'unet_Jaccard']):
        df[column] = df[column].astype('category')
        df[column] = df[column].cat.reorder_categories(metrics_order, ordered = True)
        df.sort_values(by=column, inplace=True)
        df[column] = df[column].astype('string') 
        return df 
    
    def box_plot(self, data, x, y, color, title, width=600, height = 400):
        
        fig = px.box(   
                        
                        data,
                        x=x, y=y, 
                        color=color, 
                        title=title,
                        width=width, height=height
                        
                    )


        fig.update_layout(  
                          
                                barmode='group',yaxis_range = [0,1],
                                plot_bgcolor='white',
                                font_family="Arial",
                                font = dict(size=10),
                                title_font_family="Arial",
                                #xaxis={'categoryorder':'category ascending'}#, 'showticklabels': False, 'title': None},
                                # yaxis={'showticklabels': False, 'title':None},
                                #discomment to run unet_AFM_vs_vUnet_AFM
                                # showlegend = False,
                                # boxgroupgap=0.2, boxgap=0.7
                                
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


        fig.update_traces(marker = dict(opacity = 0))
        fig.show()
        return fig
            

