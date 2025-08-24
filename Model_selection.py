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
from sarima_garch import Sarima_Garch_Model
from Data_Handler import Data_Handler


class GARCH_Model():
    def __init__(self, data: pd.DataFrame,p: int = 1,q: int = 1,):
        """
        Initializes the GARCH model.\n
        data: The input time series data.\n
        p: The order of the GARCH model (default is 1).\n
        q: The order of the GARCH model (default is 1).\n
        """
        self.data = data
        self.p = p
        self.q = q
        self.model_fit = None
        self.model = arch_model(self.data, p = self.p,q = self.q)
        self.predicted_values = None
        return self.model
        

    def fit(self,disp : str = 'off'):
        """
        Fits the GARCH model to the data.\n
        """
        self.model_fit = self.model.fit(disp = disp)
        return self.model_fit
    
    def predict(self,steps: int = None):
        """
        Predict future values using the fitted GARCH model.\n
        steps: The number of steps to forecast (default is None).\n
        """
        self.predicted_values = self.model_fit.forecast(horizon = steps)
        return self.predicted_value
    
    def model_summary(self):
        """
        Print the summary of the fitted GARCH model.
        """
        print(self.model_fit.summary())
        return self.model_fit.summary()
    
    def plot_predictions(self):
        """
        Plot the predictions of the fitted GARCH model.
        """
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.predicted_value.index, y=self.predicted_value, mode='lines', name='GARCH Predictions'))
        fig.add_trace(go.Scatter(x=self.predicted_value.index, y=[self.forecasted_mean]*len(self.predicted_value), mode='lines', name='GARCH Forecasted Mean'))
        fig.update_layout(
            title="GARCH Rolling Predictions vs Actual Data",
            xaxis_title="datetime",
            yaxis_title="Value",
            hovermode="x unified"
        )
        fig.add_trace(go.Scatter(x=self.data.index, y=self.data, mode='lines', name='Actual'))
        fig.add_trace(go.Scatter(x=self.data.index, y=self.data.mean().repeat(len(self.data)), mode='lines', name='Actual data mean'))

        fig.show()


class Model_Selection(Data_Handler, Sarima_Garch_Model, GARCH_Model):
    def __init__(self, data: pd.DataFrame,
                  model: Literal['SARIMA', 'SARIMAX', 'SARIMA-GARCH', 'LSTM', 'TACTIS-2'],Pred_steps: int = 24):
        self.original_data = data
        self.data = data
        self.model = model
        self.model_instance = None
        self.model_fit = None
        self.predicted_Values = None
        self.forecasted_mean = None
        Handler = Data_Handler(self.data)

        if model == 'SARIMA':
            steps = ["Aggregating data", "Smoothing data"]
            for step in tqdm(steps, desc="SARIMA Model Preparation", unit="step"):
                if step == "Aggregating data":
                    print("Running: Aggregating data...")
                    self.data = Handler.Data_aggregation('sum')
                elif step == "Smoothing data":
                    print("Running: Smoothing data...")
                    self.data = Handler.Data_smoothing(smoothing=True, smoothing_window=10)
                time.sleep(0.2)  # Simulate processing time
            # self.model_instance = SARIMA_Model(data)

        elif model == 'SARIMAX':
            steps = ["Aggregating data", "Smoothing data"]
            for step in tqdm(steps, desc="SARIMAX Model Preparation", unit="step"):
                if step == "Aggregating data":
                    print("Running: Aggregating data...")
                    self.data = Handler.Data_aggregation('sum')
                elif step == "Smoothing data":
                    print("Running: Smoothing data...")
                    self.data = Handler.Data_smoothing(smoothing=True, smoothing_window=10)
                time.sleep(0.2)
            # self.model_instance = SARIMAX_Model(data, exogenous=True)

        elif model == 'SARIMA-GARCH':
            steps = ["Aggregating data", "Smoothing data", "Fitting SARIMA-GARCH model", "Predicting values"]
            for step in tqdm(steps, desc="SARIMA-GARCH Model Preparation", unit="step"):
                if step == "Aggregating data":
                    print("Running: Aggregating data...")
                    self.data = Handler.Data_aggregation('sum')
                elif step == "Smoothing data":
                    print("Running: Smoothing data...")
                    self.data = Handler.Data_smoothing(smoothing=True, smoothing_window=10)
                elif step == "Fitting SARIMA-GARCH model":
                    print("Running: Fitting SARIMA-GARCH model...")
                    self.model_instance = Sarima_Garch_Model(self.data)
                    self.model_fit = self.model_instance.fit(arima_order=(2, 0, 2), seasonal_order=(1, 1, 1, 24), garch_p=2, garch_q=2)
                elif step == "Predicting values":
                    print("Running: Predicting values...")
                    self.predicted_Values = self.model_instance.predict(sarima_horizon=Pred_steps, garch_horizon=Pred_steps)
                time.sleep(0.2)

        elif model == 'LSTM':
            steps = ["Aggregating data", "Smoothing data", "Scaling data"]
            for step in tqdm(steps, desc="LSTM Model Preparation", unit="step"):
                if step == "Aggregating data":
                    print("Running: Aggregating data...")
                    self.data = Handler.Data_aggregation('sum')
                elif step == "Smoothing data":
                    print("Running: Smoothing data...")
                    self.data = Handler.Data_smoothing(smoothing=True, smoothing_window=10)
                elif step == "Scaling data":
                    print("Running: Scaling data...")
                    self.data = Handler.scaling(scaler='standard')
                time.sleep(0.2)
            # self.model_instance = LSTM_Model(data)

        elif model == 'TACTIS-2':
            steps = ["Aggregating data"]
            for step in tqdm(steps, desc="TACTIS-2 Model Preparation", unit="step"):
                if step == "Aggregating data":
                    print("Running: Aggregating data...")
                    self.data = Handler.Data_aggregation('sum')
                time.sleep(0.2)
            # self.model_instance = TACTIS2_Model(data)