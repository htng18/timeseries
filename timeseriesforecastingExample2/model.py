import pandas as pd
import numpy as np
import lightgbm as lgb
from prophet import Prophet
from statsmodels.tsa.seasonal import STL
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import skew
import optuna 
from optuna.samplers import TPESampler, RandomSampler
from utilities import FeatureEngineering


class lightGBM_predictor(object):
    '''
       The method is to use lightGBM regressor to the 
       next step. The proceeding steps are predicted using
       the regressor recursively. The general procedures are
       provided below:
       1. Decompose the time-series into the components (trend,
           seasonal and residuals).
       2. Split each component in three parts: train, dev and test.
       3. Generate the additional features of the time-series.
       4. Tune the lightGBM-regressor model using optuna with the
          lowest mean squared error between the train and dev data.
       5. Generate the prediction for each component using the best
          hyperparameters.
       6. Sum the three components into final predictions.
    '''
    def __init__(self, data, num_iteration, predict_period, period, window, model_name, n_jobs):
        self.data = data
        self.n_iter = num_iteration
        self.predict_period = predict_period
        self.period = period
        self.window = window
        res = STL(data, period=period).fit()
        self.trend = res.trend
        self.seasonal = res.seasonal
        self.residual = res.resid
        self.model_name = model_name
        self.n_jobs = n_jobs

    def result(self, input_data, target):
        scaler = MinMaxScaler(feature_range=(0,1))
        train, dev = lightGBM_predictor.train_test_split(input_data, predict_period=self.predict_period*2)
        dev, test = lightGBM_predictor.train_test_split(dev, predict_period=self.predict_period)
        train = train.to_frame()
        train.columns = [target]
        train, _, _  = lightGBM_predictor.remove_outilers(train, target)

        train = pd.DataFrame(scaler.fit_transform(train.values).flatten(), 
                            columns=[target])

        feature_eng = FeatureEngineering(target, lag=2, num_lag=8, window=5, 
                                        num_window=10, num_alpha=4)

        train = feature_eng.featuregeneration(train)
        train_X, train_y = lightGBM_predictor.Xy_generation(train, self.window, target)

        study = optuna.create_study(study_name=self.model_name, direction="minimize", 
                                    sampler=RandomSampler(seed=42),
                                    pruner=optuna.pruners.MedianPruner())
        func = lambda trial: lightGBM_predictor.objective(trial, train_X, train_y, dev, scaler, 
                                    self.predict_period, feature_eng, target, self.n_jobs, self.window)
        study.optimize(func, n_trials=self.n_iter, n_jobs=self.n_jobs)
        best_params = study.best_params

        dev_y = scaler.transform(dev.values.reshape(-1,1)).flatten()
        y = np.append(train_y, dev_y)
        y = pd.DataFrame(y, columns=[target])
        y = feature_eng.featuregeneration(y)
        X, y = lightGBM_predictor.Xy_generation(y, self.window, target)

        best_model = lgb.LGBMRegressor(n_jobs=self.n_jobs, random_state=21, **best_params)
        best_model.fit(X, y)

        pred_period = test.shape[0]

        pred = lightGBM_predictor.prediction(best_model, X, y, scaler, self.predict_period, 
                        feature_eng, target, self.window)

        return pred, best_params
    
    def lightGBM_optuna(self):
        params_dict = {}
        pred = {}
        decomposition = {'trend':self.trend, 'seasonal':self.seasonal, 'residual':self.residual}
        total_prediction = 0
        for decomp in decomposition.keys():
            pred[decomp], params_dict[decomp] = self.result(decomposition[decomp], decomp)
            total_prediction += pred[decomp]
        return total_prediction, params_dict

        
    @staticmethod
    def train_test_split(data, predict_period, split=None):
        if split is not None:
            split_size = int(split*data.shape[0])
        else:
            split_size = data.shape[0] - predict_period
        train = data[:split_size]
        test = data[split_size:]
        return train, test
    
    def Xy_generation(data, window, target):
        X = data.shift(window).dropna().values
        y = data[target][window:].values
        return X, y
    
    @staticmethod
    def remove_outilers(data, target):
        skewness = skew(data[target].values)
        if skewness > 0 and abs(skewness) > 1:
            cap = np.percentile(data[target].values, 92.5)
            floor = np.percentile(data[target].values, 2.5)
        elif skewness < 0 and abs(skewness) > 1:
            cap = np.percentile(data[target].values, 97.5)
            floor = np.percentile(data[target].values, 7.5)
        else:
            cap = np.percentile(data[target].values, 95)
            floor = np.percentile(data[target].values, 5)
        data[data[target]>=cap] = np.nan
        data[data[target]<=floor] = np.nan
        data = data.fillna(method='ffill')
        return data, cap, floor
    
    @staticmethod
    def objective(trial, X, y, dev, scaler, pred_period, feature_eng, target, n_jobs, window):
        params={'n_estimators':trial.suggest_int('n_estimators',100,1000,step=10),
                    'max_depth':trial.suggest_int('max_depth',3,12),
                    'num_leaves':trial.suggest_int('num_leaves',20,3000,step=10),
                    'min_data_in_leaf':trial.suggest_int('min_data_in_leaf',100,200,step=5),
                    'learning_rate':trial.suggest_float('learning_rate',0.01,3,step=0.05)}
        model = lgb.LGBMRegressor(n_jobs=n_jobs, random_state=21, **params)
        model.fit(X, y)
        pred = lightGBM_predictor.prediction(model, X, y, scaler, pred_period, feature_eng, target, window)
        return mean_squared_error(dev.values, pred)
    

    @staticmethod
    def prediction(model, X, y, scaler, pred_period, feature_eng, target, window):
        pred = []
        for i in range(pred_period):
            y = np.append(y, model.predict(X[-1,:].reshape(1, -1)))
            pred.append(y[-1])
            y = pd.DataFrame(y, columns=[target])
            y = feature_eng.featuregeneration(y)
            X, y = lightGBM_predictor.Xy_generation(y, window, target)
        pred = scaler.inverse_transform(np.array(pred).reshape(-1,1))
        return pred


class prophet_predictor(object):
    '''
        The method is to use prophet for predictions. 
        The general procedures are
        provided below:
        1. Split the data in three parts: train, dev and test.
        2. Generate the additional features of the time-series.
        3. Tune the prophet model using optuna with the
           lowest mean squared error between the train and dev data.
        4. Generate the prediction for each component using the best
           hyperparameters.    
    '''
    def __init__(self, data, num_iteration, predict_period, periodicity, model_name, n_jobs):
        self.data = data
        self.n_iter = num_iteration
        self.predict_period = predict_period
        self.periodicity = periodicity
        self.model_name = model_name
        self.n_jobs = n_jobs


    def prophet_optuna(self):
        frequency = {'Yearly':'Y', 'Monthly':'M', 'Daily':'D', 'Hourly':'H'}
        train, dev = prophet_predictor.train_test_split(self.data, predict_period=self.predict_period*2)
        dev, test = prophet_predictor.train_test_split(dev, predict_period=self.predict_period)
        train, cap, floor = prophet_predictor.remove_outilers(train, 'y')
        train['cap'] = cap
        train['floor'] = floor
        dev['cap'] = cap
        dev['floor'] = floor
        test['cap'] = cap
        test['floor'] = floor

        study = optuna.create_study(study_name=self.model_name, direction="minimize", 
                                    sampler=TPESampler(seed=42),
                                    pruner=optuna.pruners.MedianPruner())
        func = lambda trial: prophet_predictor.objective(trial, train, test, self.predict_period, 
                                                         cap, floor, frequency[self.periodicity])
        study.optimize(func, n_trials=self.n_iter, n_jobs=self.n_jobs)
        best_params = study.best_params

        train = pd.concat([train, dev], axis=0)

        best_model = Prophet(**best_params)
        best_model.fit(train)
        future= best_model.make_future_dataframe(periods=self.predict_period*3, freq=frequency[self.periodicity])
        future['cap'] = cap
        future['floor'] = floor
        forecast = best_model.predict(future)
        forecast = forecast.set_index('ds')
        test = test.set_index('ds')

        pred = []
        for i in forecast.index :
            if i in test.index:
                pred.append(forecast.loc[i]['yhat'])

        return pred, best_params

    @staticmethod
    def train_test_split(data, predict_period, split=None):
        if split is not None:
            split_size = int(split*data.shape[0])
        else:
            split_size = data.shape[0] - predict_period
        train = data[:split_size]
        test = data[split_size:]
        return train, test
    
    @staticmethod
    def remove_outilers(data, target):
        skewness = skew(data[target].values)
        if skewness > 0 and abs(skewness) > 1:
            cap = np.percentile(data[target].values, 92.5)
            floor = np.percentile(data[target].values, 2.5)
        elif skewness < 0 and abs(skewness) > 1:
            cap = np.percentile(data[target].values, 97.5)
            floor = np.percentile(data[target].values, 7.5)
        else:
            cap = np.percentile(data[target].values, 95)
            floor = np.percentile(data[target].values, 5)
        data[data[target]>=cap] = np.nan
        data[data[target]<=floor] = np.nan
        data = data.fillna(method='ffill')
        return data, cap, floor
    
    @staticmethod
    def objective(trial, train, test, predict_period, cap, floor, frequency):
        params={'changepoint_prior_scale': trial.suggest_float('changepoint_prior_scale', 0.005, 5),
            'changepoint_range': trial.suggest_float('changepoint_range', 0.8, 0.9),
            'seasonality_prior_scale': trial.suggest_float('seasonality_prior_scale', 0.1, 10),
            'holidays_prior_scale': trial.suggest_float('holidays_prior_scale', 0.1, 10),
            'seasonality_mode': trial.suggest_categorical('seasonality_mode', ['multiplicative', 'additive']),
            'growth': trial.suggest_categorical('growth', ['linear', 'logistic']),
            'weekly_seasonality': trial.suggest_int('weekly_seasonality', 5, 10),
            'yearly_seasonality': trial.suggest_int('yearly_seasonality', 1, 20)}
        model = Prophet(**params)
        model.fit(train)
        future= model.make_future_dataframe(periods=predict_period*3, freq=frequency)
        future['cap'] = cap
        future['floor'] = floor
        forecast = model.predict(future)
        forecast = forecast.set_index('ds')
        test = test.set_index('ds')
        prediction = []
        for i in forecast.index :
            if i in test.index:
                prediction.append(forecast.loc[i]['yhat'])
        return mean_squared_error(test['y'].values.flatten(), prediction)
