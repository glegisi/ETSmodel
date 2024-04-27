# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 09:38:00 2024

@author: ZChen4
"""
#%%
import pandas as pd
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

#%%
class ETS_Model(object):
    def __int__(self, data, error='add', trend=None, seasonal=None, 
                damped_trend=False, seasonal_periods=None):
        self.data = data
        assert type(self.data)==pd.core.series.Series, 'Data input should be a series'
        self.error = error
        self.trend = trend
        self.seasonal = seasonal
        self.damped_trend = damped_trend
        self.seasonal_periods = seasonal_periods
        
        self.model = ETSModel(
            endog=self.data,
            error=self.error,
            trend=self.trend,
            seasonal=self.seasonal,
            damped_trend=self.damped_trend,
            seasonal_periods=self.seasonal_periods,
        )
        self.fit = self.model.fit(disp=False)
    
    def plot(self):
        pass
        
    def summary(self):
        print(self.fit().summary())
        res_df = pd.DataFrame()
        res_df['ETS Model Name'] = self.fit().short_name
        res_df['AIC'] = self.fit().aic
        res_df['Smoothing Level'] = self.fit().alpha
        
        if self.trend!=None:
            res_df['Smoothing Trend'] = self.fit().gamma
            
        if self.seasonal!=None:
            res_df['Smoothing Seasonal'] = self.fit().beta
        
        if self.trend!=None:
            res_df['Damping Trend'] = self.fit().phi
            
        if self.seasonal!=None:
            res_df['Seasonal Periods'] = self.fit().seasonal_periods
            
        res_df.to_csv('Paramters.csv', index=False)
        
      