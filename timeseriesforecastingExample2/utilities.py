import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import skew

class FeatureEngineering(object):
    '''
       This class is to generate the features for a time-series.
       It generates the features in the following:
       1. Lags
       2. Rolling windows ('Mean', 'Max', 'Min', 'Std')
       3. Exponential weighted moving average
    
    '''
    def __init__(self, target, lag=2, num_lag=10, window=3, num_window=5, num_alpha=4):
        self.target = target
        self.lag = lag
        self.num_lag = num_lag
        self.window = window
        self.num_window = num_window
        self.num_alpha = num_alpha
        
    
    def lag_generation(self, data):
        '''
        Add the time-lagged features of the time series with column name, "target"
        Args:
          data (pandas DataFrame): the input DataFrame with the different features
          target (str): The column name of the time-series forecasting.
          lag (int): The time unit of lags for the time-lagged features.
          num_lag (int): The number of time-lag features to be added.
        Returns:
          pandas DataFrame: The dataframe with the added time-lagged features.
        '''
        lag = [i for i in range(self.lag, (self.num_lag+1)*self.lag, self.lag)]
        for i in lag:
            data['lag'+str(i)] = data[self.target].shift(i).fillna(0.0)
        column = list(data.columns)
        column.remove(self.target)
        column = column + [self.target]
        data = data[column]
        return data
    
    
    def rollingwindow_generation(self, data, metrics='mean'):
        '''
        Add the moving-average features of the time series with column name, 
        "target" with the different window size.
        Args:
          data (pandas Dataframe): the input DataFrame with the different features.
          target (str): The column name of the time-series forecasting.
          window (int): The window size of moving average.
          num_window (int): the number of moving-average features to be added, where 
                            the window size is multiple of window.
        Returns:
          pandas DataFrame: The dataframe with the added moving-average features.
        '''
        rollingwindow = [i for i in range(self.window, (self.num_window+1)*self.window, self.window)]
        if metrics=='mean':
            for i in rollingwindow:
                data['rollingwindow_'+metrics+'_'+str(i)] = data[self.target].rolling(i).mean().fillna(0.0)
        elif metrics=='std':
            for i in rollingwindow:
                data['rollingwindow_'+metrics+'_'+str(i)] = data[self.target].rolling(i).std().fillna(0.0)
        elif metrics=='max':
            for i in rollingwindow:
                data['rollingwindow_'+metrics+'_'+str(i)] = data[self.target].rolling(i).max().fillna(0.0)
        elif metrics=='min':
            for i in rollingwindow:
                data['rollingwindow_'+metrics+'_'+str(i)] = data[self.target].rolling(i).min().fillna(0.0)

        column = list(data.columns)
        column.remove(self.target)
        column = column + [self.target]
        data = data[column]
        return data

    def ewma_generation(self, data):
        '''
        Add the exponential-weighted-moving-average features of the time series with column name, 
        "target" with the different window size.
        Args:
          data (pandas Dataframe): the input DataFrame with the different features.
          target (str): The column name of the time-series forecasting.
          num_alpha (int): the number of ewma features to be added, where 
                            alpha_i = i/num_alpha.
        Returns:
          pandas DataFrame: The dataframe with the added ewma features.
        '''
        da = 1/self.num_alpha
        alpha_list = [round(da*(i+1),3) for i in range(num_alpha)]
        for i in alpha_list:
            data['ewma_'+str(i)] = data[self.target].ewm(alpha=i).mean()
        column = list(data.columns)
        column.remove(self.target)
        column = column + [self.target]
        data = data[column]
        return data
        
    
    def featuregeneration(self, data):
        size = data.shape[0]
        data = self.lag_generation(data)
        for i in ['mean','std','max','min']:
            data = self.rollingwindow_generation(data, metrics=i)
        return data 
    
class metrics_computation(object):
    '''
       This class is to compute the metrics of the prediction 
       of time-series. It provides the following metrics:
       1. SMAPE
       2. MASE
       3. MSE
       4. MAE
    
    '''
    def __init__(self, test, insample, pred, periodicity):
        self.test = test
        self.insample = insample
        self.pred = pred
        self.periodicity = periodicity


    def compute_metrics(self):
        SMAPE = round(self.smape(),2)
        MASE = round(self.mase(),2)
        MSE = round(mean_squared_error(self.test, self.pred),2)
        MAE = round(mean_absolute_error(self.test, self.pred),2)
        return SMAPE, MASE, MSE, MAE


    def smape(self):
        '''
        Compute SMAPE
        '''
        return 100/len(self.test) * np.sum(2 * np.abs(self.test - self.pred) 
                                           / (np.abs(self.test) + np.abs(self.pred)))

    def mase(self):
        '''
        Compute MASE
        '''
        period = {"Yearly":1, "Quarterly":4, "Monthly":12, 
                       "Weekly":1, "Daily":1, "Hourly":24}
        m = period[self.periodicity]
        d = np.nansum(np.abs(self.insample[:-m]
                             -np.roll(self.insample, -m)[:-m]))/(len(self.insample[:-m]))
        return np.sum(np.abs(self.test - self.pred))/d/len(self.test)