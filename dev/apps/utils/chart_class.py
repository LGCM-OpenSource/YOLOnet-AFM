import plotly.express as px
import pandas as pd
class Charts:
    def __init__(self, width=600, height=400, color_discrete_sequence=px.colors.qualitative.Plotly):
        self.width = width
        self.height = height
        self.color_discrete_sequence = color_discrete_sequence

    def _apply_layout(self, fig, title, x_label, y_label, show_x_labels, show_y_labels):
        fig.update_layout(
            title=title,
            width=self.width,
            height=self.height,
            plot_bgcolor='white',
            font_family="Arial",
            font=dict(size=10),
            title_font_family="Arial",
            xaxis=dict(
                showticklabels=show_x_labels,
                title=x_label if show_x_labels else None,
                mirror=True,
                linecolor='black',
                ticks='outside',
                showline=True,
                gridcolor='lightgrey'
            ),
            yaxis=dict(
                showticklabels=show_y_labels,
                title=y_label if show_y_labels else None,
                mirror=True,
                linecolor='black',
                ticks='outside',
                showline=True,
                gridcolor='lightgrey',
                tick0=0, 
                dtick=0.1
            )
        )

    def box_plot(self, data, x, y, color=None, title='', x_label='', y_label='', show_x_labels=True, show_y_labels=True):
        fig = px.box(
            data_frame=data,
            x=x,
            y=y,
            color=color,
            color_discrete_sequence=self.color_discrete_sequence
        )
        self._apply_layout(fig, title, x_label, y_label, show_x_labels, show_y_labels)
        fig.show()
        return fig

    def violin_plot(self, data, x, y, color=None, title='', x_label='', y_label='', show_x_labels=True, show_y_labels=True, points='all'):
        fig = px.violin(
            data_frame=data,
            x=x,
            y=y,
            color=color,
            box=True,
            points='all',
            color_discrete_sequence=self.color_discrete_sequence
        )
        self._apply_layout(fig, title, x_label, y_label, show_x_labels, show_y_labels)
        fig.show()
        return fig

    def line_plot(self, data, x, y, color=None, title='', x_label='', y_label='', show_x_labels=True, show_y_labels=True):
        fig = px.line(
            data_frame=data,
            x=x,
            y=y,
            color=color,
            markers=True,
            color_discrete_sequence=self.color_discrete_sequence
        )
        self._apply_layout(fig, title, x_label, y_label, show_x_labels, show_y_labels)
        fig.show()
        return fig



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
            
    def apply_melt(self, df, id_vars = ['Model', 'Process Date'], value_vars=[	'Precision', 'Recall',	'F1', 'Dice'] , var_name= 'metrics' , value_name='scores'):
         return pd.melt(df, id_vars=id_vars, value_vars=value_vars , var_name= var_name , value_name=value_name)   
    
    def ordenate_metrics(self, df, column, metrics_order = ['Precision'  ,    'Recall'  ,  'F1'  , 'Dice']):
        df[column] = df[column].astype('category')
        df[column] = df[column].cat.reorder_categories(metrics_order, ordered = True)
        df.sort_values(by=column, inplace=True)
        df[column] = df[column].astype('string') 
        return df 