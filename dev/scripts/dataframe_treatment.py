import pandas as pd
import numpy as np 
import traceback

'''
CLASS               LINE

DataFrameTrat        12
PreProcessDataframe  120
'''

class DataFrameTrat:
    """
    A class used for processing and manipulating DataFrames.

    ...

    Attributes
    ----------
    self : DataFrameTrat
        An instance of the DataFrameTrat class.
    df_path : str
        The path to the CSV file containing the DataFrame.
    target : str
        The name of the target column in the DataFrame.

    Methods
    -------
    df : pandas.DataFrame
        Returns the loaded DataFrame from the given path.
    clean_target(df)
        Cleans and maps values in the target column to numeric values.
    normalize_columns(df, column_name)
        Normalizes the values in the specified column.
    create_channel_by_df(df, column, dimension)
        Creates a NumPy array from a DataFrame column and reshapes it to the specified dimension.
    """
    def __init__(self, df_path):
        """
        Constructs an instance of the DataFrameTrat class.

        Parameters
        ----------
        df_path : str
            The path to the CSV file containing the DataFrame.
        """
        self.df_path = df_path 
        self.target = 'Generic Segmentation'
      
    @property
    def df(self):
        """
        Returns the loaded DataFrame from the given path.

        Returns
        -------
        pandas.DataFrame
            The loaded DataFrame.
        """
        return pd.read_csv(self.df_path, sep='\t', index_col=0)    
    
    def clean_target(self, df):
        """
        Cleans and maps values in the target column to numeric values.

        Parameters
        ----------
        df : pandas.DataFrame
            The input DataFrame.

        Returns
        -------
        pandas.DataFrame
            The DataFrame with cleaned and mapped target values.
        """
        segment = {'Cytoplasm': 0 ,'Nucleus': 1, 'Pericellular': 0, float('nan'): 0}
        df[ self.target] = df[ self.target].replace(segment)
        return df

    def normalize_columns(self, df, column_name):
        """
        Normalizes the values in the specified column.

        Parameters
        ----------
        df : pandas.DataFrame
            The input DataFrame.
        column_name : str
            The name of the column to be normalized.

        Returns
        -------
        pandas.DataFrame
            The DataFrame with normalized values in the specified column.
        """
        df[column_name] = ((df[column_name] - min(df[column_name])) / (max(df[column_name]) - min(df[column_name])))
        return df

    def create_channel_by_df(self, df, column, dimension):
        """
        Creates a NumPy array from a DataFrame column and reshapes it to the specified dimension.

        Parameters
        ----------
        df : pandas.DataFrame
            The input DataFrame.
        column : str
            The name of the column to create the channel from.
        dimension : tuple
            The desired dimensions of the resulting NumPy array in (height, width) format.

        Returns
        -------
        numpy.ndarray
            The created NumPy array.
        """
        return np.array(df[column]).reshape(dimension[0], dimension[1])
    

class PreProcessDataframe:
    def __init__(self, df_path):
        self.df_afm = DataFrameTrat(df_path)
        self.features = ['Process Date', 
                                       'Height Position',
                                       'Planned Height',
                                       'Norm Height',
                                       'MaxPosition_F0500pN',
                                       'YM_Fmax0500pN',
                                       'Generic Segmentation']

    def minMaxscaler(self, df, column_name):
        df[column_name] = ((df[column_name] - min(df[column_name])) / (max(df[column_name]) - min(df[column_name])))
        return df

    def create_norm_height(self, df, column_name):
            df[column_name] = ((df['Planned Height'] - min(df['Planned Height'])) / (max(df['Planned Height']) - min(df['Planned Height'])))
            return df

    def create_height_position(self, df, column_name):
            df[column_name] = df['Planned Height'].rank(ascending=False)
            return df

    def standardScale(self, df,column_name):
        df[column_name] = ((df[column_name] - df[column_name].mean() ) / df[column_name].std() )
        return df
    
    def run_preprocess_pixel_segmentation(self, save_path):
        usefull_data = self.df_afm.df
        #creating the two derivatives, if the dataframe did not have
        if 'Norm Height' not in usefull_data.columns:
            usefull_data = self.create_norm_height(usefull_data, 'Norm Height')

        if 'Height Position' not in usefull_data.columns:
            usefull_data = self.create_height_position(usefull_data, 'Height Position')
            
        usefull_data = usefull_data[self.features].copy()

        remove_list = ['Generic Segmentation', 'Planned Height', 'Norm Height','Height Position', 'Process Date']

        #se a planilha tiver mais de uma imagem -- normalizar uma a uma
        normalized_df = pd.DataFrame()

        for cel in usefull_data['Process Date'].unique():
            try:
                #Pegando uma célula do dataset de treino
                df_cell = usefull_data.loc[usefull_data['Process Date']== cel]
                df_cell = df_cell[self.features].copy()
                
                #retirar colunas que não devem ser normalizadas
                X = df_cell.drop(remove_list, axis=1)
                cols = X.columns.tolist()
                
                #normalização
                norm_list = []
                for col in cols:
                    X = self.standardScale(X, col)
                df_normalize = pd.DataFrame(X, columns=cols, index=df_cell.index.values)
                
                #concatenando as colunas normalizadas com as que não foram normalizadas
                df = df_normalize.join(df_cell[remove_list])
                normalized_df = pd.concat([normalized_df, df], ignore_index=True) 
                
                normalized_df = self.df_afm.clean_target(normalized_df)
                
                normalized_df.to_csv(save_path, sep='\t')
            
            except Exception:
                traceback.print_exc()