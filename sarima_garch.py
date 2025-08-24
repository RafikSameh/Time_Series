# Import Libraries
import pandas as pd
import numpy as np
from datetime import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import het_arch
from arch import arch_model
from arch.univariate import GARCH, ConstantMean, ARX, HARX, EGARCH
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import root_mean_squared_error as RMSE
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import plotly.graph_objects as go
from typing import Literal
from tqdm import tqdm
import time


class Sarima_Garch_Model():
    """
    This Class is for a model that is combination between SARIMA model for values and mean predictions
    and GARCH model for volatility and variance estimation.
    It allows for rolling predictions and evaluation of the model's performance.\n
    stationarity_test: Performs stationarity test on the data.\n
    fit_predict_rolling: Fits the SARIMA and GARCH models using rolling predictions.\n
    model_evaluation: Evaluates the performance of the fitted models.\n
    plot_results: Plots the actual vs predicted values.\n
    """
    def __init__(self, data: pd.DataFrame):
        """
        Initializes the Sarima_Garch_Model with data.\n
        data: Dataframe that will be used for modeling. (only one feature is allowed and indexed by time)\n
        This Class is for a model that is combination between SARIMA model for values and mean predictions
        and GARCH model for volatility and variance estimation.
        It allows for rolling predictions and evaluation of the model's performance.\n
        stationarity_test: Performs stationarity test on the data.\n
        fit_predict_rolling: Fits the SARIMA and GARCH models using rolling predictions.\n
        model_evaluation: Evaluates the performance of the fitted models.\n
        plot_results: Plots the actual vs predicted values.\n
        """
        self.data = data
        # Smoothing
        self.smoothing = None
        # Rolling predictions
        self.sarima_rolling_predictions = None
        self.garch_rolling_predictions = None
        self.predicted_values = None
        self.forecasted_mean = 0
        # Models
        self.sarima_model = None
        self.garch_model = None
        # Window for rolling
        self.train_window = None
        # Fitted models
        self.sarima_fit = None
        self.garch_fit = None
        # Predictions without rolling
        self.sarima_forecast = None
        self.garch_forecast = None
        self.combined_forecast = None


    '''def data_preprocessing(self,smoothing: bool = None, smoothing_window: int = None, Aggregation: Literal['mean', 'sum'] = 'sum'):
        """
        Preprocess the data by applying smoothing if specified.
        smoothing: Boolean indicating whether to apply smoothing to the data.\n
        smoothing_window: Integer representing the window size for smoothing.\n
        Aggregation: String indicating the type of aggregation to apply (mean or sum).\n
        """
        self.data.index = pd.to_datetime(self.data.index)
        self.smoothing = smoothing
        
        if self.smoothing:
            self.data = self.data.rolling(smoothing_window).mean()

        if Aggregation == 'mean':
            self.data = self.data.resample('h').mean()
        else:
            self.data = self.data.resample('h').sum()'''

    
    

    def fit(self, arima_order, seasonal_order, garch_p, garch_q):
        self.sarima_model = SARIMAX(self.data, order=arima_order, seasonal_order=seasonal_order)
        self.sarima_fit = self.sarima_model.fit(disp=False)
        self.garch_model = arch_model(self.data, vol='Garch', p=garch_p, q=garch_q, rescale=False)
        self.garch_fit = self.garch_model.fit(disp='off')

    def predict(self, sarima_horizon, garch_horizon):
        self.sarima_forecast = self.sarima_fit.forecast(steps=sarima_horizon)
        self.garch_forecast = self.garch_fit.forecast(horizon=garch_horizon)
        self.garch_forecast = np.sqrt(self.garch_forecast.variance.values[-1, :])
        self.forecasted_mean = self.sarima_forecast.mean()
        # Combine SARIMA and GARCH forecasts
        combined_forecast = []
        value = None
        for i in range(len(self.sarima_forecast)):
            if self.sarima_forecast.iloc[i] >= self.forecasted_mean:
                value = self.forecasted_mean + self.garch_forecast[i]
            else:
                value =  self.forecasted_mean - self.garch_forecast[i]
            combined_forecast.append(value)

        self.combined_forecast = pd.Series(combined_forecast, index=self.sarima_forecast.index, name='predicted_Values')
        return self.combined_forecast

    def fit_predict_rolling(self, arima_order: tuple = (2,0,2), seasonal_order: tuple = (1,1,1,24), garch_p: int = 1,
                  garch_q: int = 1, training_window: int = 24, sarima_pred_steps: int = 1, rolling_step_sarima: int = 1,
                  garch_pred_steps: int = 1, rolling_step_garch: int = 1):
        """
        Fits the SARIMA and GARCH models using rolling predictions.\n
        arima_order: Order of the SARIMA model.\n
        seasonal_order: Seasonal order of the SARIMA model.\n
        garch_p: Order of the GARCH model (lagged variance).\n
        garch_q: Order of the ARCH model (lagged error).\n
        training_window: Size of the training window.\n
        sarima_pred_steps: Number of steps to predict with SARIMA.\n
        garch_pred_steps: Number of steps to predict with GARCH.\n
        rolling_step_sarima: Step size for rolling predictions with SARIMA.\n
        rolling_step_garch: Step size for rolling predictions with GARCH.\n
        """
        #######################################################################
        ######                      SARIMA                               ######
        #######################################################################
        # Fit the SARIMA model and GARCH model using rolling predictions
        self.sarima_rolling_predictions = []
        self.train_window = training_window # 5 days
        forecast_horizon_SARIMA = sarima_pred_steps # 1 hour
        step_SARIMA = rolling_step_sarima # roll by 1 hour


        # Ensure enough data for at least one window and forecast
        if len(self.data) < self.train_window + forecast_horizon_SARIMA:
            print("Not enough data for the specified window and horizon.")
        else:
            # Determine the number of rolling windows
            num_windows = (len(self.data) - self.train_window - forecast_horizon_SARIMA) // step_SARIMA + 1

            for i in range(num_windows):
                # Define the training and testing slices for the current window
                start_idx = i * step_SARIMA
                end_idx_train = start_idx + self.train_window
                end_idx_test = end_idx_train + forecast_horizon_SARIMA

                current_train = self.data.iloc[start_idx:end_idx_train]
                current_test_index = self.data.iloc[end_idx_train:end_idx_test].index


                # Fit the SARIMA model
                self.sarima_model = SARIMAX(current_train, order=arima_order, seasonal_order=seasonal_order) # Simplified SARIMA order
                sarima_model_fit = self.sarima_model.fit(disp=False, enforce_stationarity=False, enforce_invertibility=False) # Added convergence arguments
                forecasted_sarima_values = sarima_model_fit.get_forecast(steps=forecast_horizon_SARIMA).predicted_mean
                # forecasted_mean = forecasted.mean() # This line is not needed here

                # Store the forecasts with the correct index
                self.sarima_rolling_predictions.append(pd.Series(forecasted_sarima_values, index=current_test_index))

        # Concatenate the list of Series into a single Series
        self.sarima_rolling_predictions = pd.concat(self.sarima_rolling_predictions)
        self.forecasted_mean = self.sarima_rolling_predictions.mean()
        print("SARIMA Rolling Prediction is completed successfully")

        #######################################################################
        ########################       GARCH         ########################## 
        #######################################################################
        self.garch_rolling_predictions = []
        forecast_horizon_GARCH = garch_pred_steps
        step_GARCH = rolling_step_garch
        # Ensure enough data for at least one window and forecast
        if len(self.data) < self.train_window + forecast_horizon_GARCH:
            print("Not enough data for the specified window and horizon.")
        else:
            # Determine the number of rolling windows
            '''if self.smoothing:
                num_windows = (len(self.data) - self.train_window - forecast_horizon_GARCH) + 1
            else:'''
            num_windows = (len(self.data) - self.train_window - forecast_horizon_GARCH) // step_GARCH + 1

            for i in range(num_windows):
                # Define the training and testing slices for the current window
                start_idx = i * step_GARCH
                end_idx_train = start_idx + self.train_window
                end_idx_test = end_idx_train + forecast_horizon_GARCH  
                # If smoothing is applied, drop NaN values in the training set 
                if self.smoothing:
                    current_train = self.data.iloc[start_idx:end_idx_train].dropna()
                    current_test_index = self.data.iloc[end_idx_train:end_idx_test].index
                else:
                    current_train = self.data.iloc[start_idx:end_idx_train]
                    current_test_index = self.data.iloc[end_idx_train:end_idx_test].index

                # Fit the GARCH model
                self.garch_model = arch_model(current_train,vol="GARCH", p=garch_p, q=garch_q,rescale=False)
                model_fit = self.garch_model.fit(disp="off")

                # Forecast volatility for the horizon
                pred = model_fit.forecast(horizon=forecast_horizon_GARCH)
                volatility_forecast = np.sqrt(pred.variance.values[-1, :])

                # Store the forecasts with the correct index
                self.garch_rolling_predictions.append(pd.Series(volatility_forecast, index=current_test_index))

        # Concatenate the list of Series into a single Series
        self.garch_rolling_predictions = pd.concat(self.garch_rolling_predictions)
        print("GARCH Rolling Prediction is completed successfully")
        #######################################################################
        ########################       Combined      ##########################
        #######################################################################

        self.predicted_values = []
        value = None
        for i in range(len(self.sarima_rolling_predictions)):
            if self.sarima_rolling_predictions.iloc[i] >= self.forecasted_mean:
                value = self.forecasted_mean + self.garch_rolling_predictions.iloc[i]
            else:
                value =  self.forecasted_mean - self.garch_rolling_predictions.iloc[i]
            self.predicted_values.append(value)

        self.predicted_values = pd.Series(self.predicted_values, index=self.sarima_rolling_predictions.index,name = 'predicted_Values')
        print("Combined Rolling Prediction is completed successfully")


    def evaluation(self,rolling: bool = True ,model_to_evaluate: Literal['sarima', 'garch', 'combined'] = 'combined',
                    eval_metric: Literal['mse', 'mae', 'mape','rmse'] = 'mape',
                    start_index: int = None,end_index: int = None):
        """
        Evaluate the specified model using the chosen metric.\n
        model_to_evaluate: The model to evaluate ['sarima', 'garch', 'combined'].\n
        eval_metric: The evaluation metric to use ['mse', 'mae', 'mape', 'rmse'].\n
        start_index: start of single forecast
        end_index: end of single forecast
        """
        # Evaluate the specified model
        if rolling:
            if model_to_evaluate == 'sarima':
                if eval_metric == 'mse':
                    return MSE(self.data[self.train_window:],self.sarima_rolling_predictions)
                elif eval_metric == 'mae':
                    return MAE(self.data[self.train_window:],self.sarima_rolling_predictions)
                elif eval_metric == 'rmse':
                    return RMSE(self.data[self.train_window:],self.sarima_rolling_predictions)
                else:
                    return MAPE(self.data[self.train_window:],self.sarima_rolling_predictions)
            elif model_to_evaluate == 'garch':
                if eval_metric == 'mse':
                    return MSE(self.data[self.train_window:],self.garch_rolling_predictions)
                elif eval_metric == 'mae':
                    return MAE(self.data[self.train_window:],self.garch_rolling_predictions)
                elif eval_metric == 'rmse':
                    return RMSE(self.data[self.train_window:],self.garch_rolling_predictions)
                else:
                    return MAPE(self.data[self.train_window:],self.garch_rolling_predictions)
            elif model_to_evaluate == 'combined':
                if eval_metric == 'mse':
                    return MSE(self.data[self.train_window:],self.predicted_values)
                elif eval_metric == 'mae':
                    return MAE(self.data[self.train_window:],self.predicted_values)
                elif eval_metric == 'rmse':
                    return RMSE(self.data[self.train_window:],self.predicted_values)
                else:
                    return MAPE(self.data[self.train_window:],self.predicted_values)
        else:
            if model_to_evaluate == 'sarima':
                if eval_metric == 'mse':
                    return MSE(self.data[start_index:end_index],self.sarima_forecast)
                elif eval_metric == 'mae':
                    return MAE(self.data[start_index:end_index],self.sarima_forecast)
                elif eval_metric == 'rmse':
                    return RMSE(self.data[start_index:end_index],self.sarima_forecast)
                else:
                    return MAPE(self.data[start_index:end_index],self.sarima_forecast)
            elif model_to_evaluate == 'garch':
                if eval_metric == 'mse':
                    return MSE(self.data[start_index:end_index],self.garch_forecast)
                elif eval_metric == 'mae':
                    return MAE(self.data[start_index:end_index],self.garch_forecast)
                elif eval_metric == 'rmse':
                    return RMSE(self.data[start_index:end_index],self.garch_forecast)
                else:
                    return MAPE(self.data[start_index:end_index],self.garch_forecast)
            elif model_to_evaluate == 'combined':
                if eval_metric == 'mse':
                    return MSE(self.data[start_index:end_index],self.combined_forecast)
                elif eval_metric == 'mae':
                    return MAE(self.data[start_index:end_index],self.combined_forecast)
                elif eval_metric == 'rmse':
                    return RMSE(self.data[start_index:end_index],self.combined_forecast)
                else:
                    return MAPE(self.data[start_index:end_index],self.combined_forecast)


    def plot_predictions(self, model_to_plot: Literal['sarima', 'garch', 'combined'] = 'combined'):
        """
        Plot the predictions of the specified model.\n
        model_to_plot: The model to plot ['sarima', 'garch', 'combined'].\n
        """
        fig = go.Figure()
        if model_to_plot == 'sarima':
            fig.add_trace(go.Scatter(x=self.sarima_rolling_predictions.index, y=self.sarima_rolling_predictions, mode='lines', name='SARIMA Predictions'))
            fig.add_trace(go.Scatter(x=self.sarima_rolling_predictions.index, y=[self.forecasted_mean]*len(self.sarima_rolling_predictions), mode='lines', name='SARIMA Forecasted Mean'))
            fig.update_layout(
                title="SARIMA Rolling Predictions vs Actual Data",
                xaxis_title="datetime",
                yaxis_title="Value",
                hovermode="x unified"
            )

        elif model_to_plot == 'garch':
            fig.add_trace(go.Scatter(x=self.garch_rolling_predictions.index, y=self.garch_rolling_predictions, mode='lines', name='GARCH Predictions'))
            fig.add_trace(go.Scatter(x=self.garch_rolling_predictions.index, y=[self.forecasted_mean]*len(self.garch_rolling_predictions), mode='lines', name='GARCH Forecasted Mean'))
            fig.update_layout(
                title="GARCH Rolling Predictions vs Actual Data",
                xaxis_title="datetime",
                yaxis_title="Value",
                hovermode="x unified"
            )

        elif model_to_plot == 'combined':
            fig.add_trace(go.Scatter(x=self.predicted_values.index, y=self.predicted_values, mode='lines', name='Combined Predictions'))
            fig.add_trace(go.Scatter(x=self.predicted_values.index, y=[self.forecasted_mean]*len(self.predicted_values), mode='lines', name='Combined Forecasted Mean'))
            fig.update_layout(
                title="SARIMA_GARCH Rolling Predictions vs Actual Data",
                xaxis_title="datetime",
                yaxis_title="Value",
                hovermode="x unified"
            )

        fig.add_trace(go.Scatter(x=self.data.index, y=self.data, mode='lines', name='Actual'))
        fig.add_trace(go.Scatter(x=self.data.index, y=self.data.mean().repeat(len(self.data)), mode='lines', name='Actual data mean'))

        fig.show()

