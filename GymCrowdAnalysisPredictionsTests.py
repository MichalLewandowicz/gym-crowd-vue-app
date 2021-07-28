# -*- coding: utf-8 -*-
"""
Software Design Masters (Artificial Intelligence), Full-Time 
Athlone Institute of Technology

author: Lee O' Connor
Student ID: A00239789
Email: a00239789@student.ait.ie

Program description
<---------------------------------------------------------->

<---------------------------------------------------------->

"""

import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn

from sklearn.metrics import accuracy_score, median_absolute_error, r2_score,mean_squared_error, explained_variance_score, max_error

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor


from sklearn.feature_selection import RFE,RFECV
from sklearn.pipeline import Pipeline

from datetime import datetime
import time

  

def get_date(series):
         return series.str.slice(8,11)
     
def buildDataStructure():
    #read in gym_data
    gym_data = pd.read_csv("GymData.csv") 
    
    
    #convert temperature column from Farenheit to Celcius
    gym_data.temperature = gym_data.temperature.apply(lambda x: int((x-32)*(5/9)))
    gym_data.rename(columns={"temperature":"temperature_celcius"}, inplace = True)
    

    #Change day of week to 1-7 instead of 0-6 for readability, 1 = Monday, 7 = Sunday
    gym_data.day_of_week = gym_data.day_of_week.apply(lambda x: x+1)
    
    #delete 'is_holidays' column, as it does not contain full information for each year 
    del gym_data['is_holiday']
    
    del gym_data['timestamp']
    del gym_data['is_start_of_semester']
    
   
    #dataframe with only the day of the month from the date
    gym_data['day_of_month'] = gym_data[['date']].apply(get_date)
    
    #build independent varables and dependent variable( number of people in the gym)
    crowds_X = gym_data.iloc[:, 2:]
    crowds_Y = gym_data.iloc[:, 0:1]
    
    
    
    #build train and test data
    crowds_X_train, crowds_X_test, crowds_Y_train, crowds_Y_test = train_test_split(crowds_X, crowds_Y, test_size = 0.20, random_state = 8)
       
    return gym_data, crowds_X_train, crowds_X_test, crowds_Y_train, crowds_Y_test

   
def getRegressionMetrics(model_name,predicted_results,time):
    crowd_mean_square_error = mean_squared_error(crowds_Y_test,predicted_results)
    crowd_r2 = r2_score(crowds_Y_test,predicted_results)
    crowd_root_mean_square_error = np.sqrt(crowd_mean_square_error)
    crowd_median_absolute_error = median_absolute_error(crowds_Y_test,predicted_results)
    crowd_max_error = max_error(crowds_Y_test,predicted_results)
    crowd_explained_variance_score =  explained_variance_score(crowds_Y_test,predicted_results)
    
    new_row = {"Model":model_name,
               "R Squared":crowd_r2,
               "Mean Square Error" :crowd_mean_square_error,
               "Root Mean Square Error": crowd_root_mean_square_error,
               "Median Absolute Error": crowd_median_absolute_error,
               "Max Error": crowd_max_error,
               "Explained Varience Score": crowd_explained_variance_score,
               "Prediction Time in seconds": time,
               "Actual Results": list(predicted_results)
               }
    return new_row



if __name__ =="__main__":
    
    gym_data, crowds_X_train, crowds_X_test, crowds_Y_train, crowds_Y_test = buildDataStructure()
    
    column_names =["Model","R Squared", "Mean Square Error","Root Mean Square Error",
                   "Median Absolute Error","Max Error", "Explained Varience Score","Prediction Time","Actual Results"]
    results_df = pd.DataFrame(columns = column_names)

    gym_models = {
        "linear Regression": LinearRegression(),
        "RFE Linear Regression": RFECV(LinearRegression()),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state = 0),
        "Decision Tree Regressor 2": DecisionTreeRegressor(splitter = "best",max_depth=10),
        "RFE Decission Tree Regressor":RFECV( DecisionTreeRegressor(random_state = 0) ),
        "KNN Regressor": KNeighborsRegressor(n_neighbors=12),
        "Random Forest Regressor": RandomForestRegressor(n_estimators = 100, random_state = 42)
        
        }
    numeric_results = pd.DataFrame(list(crowds_Y_test.number_people), columns=["Actual"])
    
    for name, model in gym_models.items():
        model.fit(crowds_X_train, crowds_Y_train.values.ravel())
        start = time.time()
        predictions = model.predict(crowds_X_test)
        predictions = predictions.astype(int)
        prediction_time = str(time.time()- start)
        results_df = results_df.append(getRegressionMetrics(name,predictions,prediction_time), ignore_index=True)
        numeric_results[name] = predictions
     
   