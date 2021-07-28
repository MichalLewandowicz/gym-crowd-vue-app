#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 11:22:22 2021

@author: grace
"""
# -*- coding: utf-8 -*-
"""
MSc Software Design with Artificial Intelligence, 
Work Placement with Software Research Institute, 
Athlone Institute of Technology.

Grace O' Brien, A00277431


Program description
<---------------------------------------------------------->
Applying different machine learning models to dataset, to determine most 
suitable for use in app to predict crowdedness in a college gym.
<---------------------------------------------------------->

"""

from sklearn.preprocessing import PolynomialFeatures

import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn


from sklearn.metrics import accuracy_score, median_absolute_error, r2_score,mean_squared_error, explained_variance_score, max_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor


from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.pipeline import Pipeline

from datetime import datetime
from sklearn.tree import export_graphviz
 
  

def getRegressionMetrics(model_name,predicted_results):
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
               "Explained Varience Score": crowd_explained_variance_score
               }
    return new_row
    
    


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
    
    #delete 'timestamp' column, as it contains the same info as 'hour' column
    del gym_data['timestamp']
    
    #It may be useful to add a column to the dataframe with day of the month details, which can be extracted from the date column
    #get characters between 8th and 11th place to extract just the day of the month, from the whole date
    def get_date(series):
        return series.str.slice(8,11) 
    #dataframe with only the day of the month from the date
    gym_data['day_of_month'] = gym_data[['date']].apply(get_date)
    
    #build independent varables and dependent variable( number of people in the gym)
    crowds_X = gym_data.iloc[:, 2:10]
    crowds_Y = gym_data.iloc[:, 0:1]
    
    #build train and test data
    crowds_X_train, crowds_X_test, crowds_Y_train, crowds_Y_test = train_test_split(crowds_X, crowds_Y, test_size = 0.20, random_state = 8)
    
# =============================================================================
    #scaler = StandardScaler()
    #scaler.fit(crowds_X_train)
    #crowds_X_train = scaler.transform(crowds_X_train)
    #crowds_X_test = scaler.transform(crowds_X_test)
    
    #scale temperature
    #scaler1 = StandardScaler()
    #scaler1.fit(crowds_X_train[:, 2:3])
    #crowds_X_train[:, 2:3] = scaler1.transform(crowds_X_train[:, 2:3])
    #crowds_X_test[:, 2:3] = scaler1.transform(crowds_X_test[:, 2:3])
    

# =============================================================================
    
    return gym_data, crowds_X_train, crowds_X_test, crowds_Y_train, crowds_Y_test

def testFeatures(gym):
    boolean_features_dates = {}
    holiday_dates = []
    weekend_dates = []
    start_dates = []
    unique_dates = []
    for index, row in gym.iterrows():
        
        date_test = row.date
        date_test = date_test.split(" ")
        date_t = date_test[0].strip()
        unique_dates.append(date_t)
        
        if row.is_holiday == 1:
            date_test = row.date
            date_test = date_test.split(" ")
            date_t = date_test[0].strip()
            holiday_dates.append(date_t)
            
        #print(f"------\nIndex: {index},\nRow: {row.number_people}")
        if row.is_weekend == 1:
            date_test = row.date
            date_test = date_test.split(" ")
            date_t = date_test[0].strip()
            weekend_dates.append(date_t)
        if row.is_start_of_semester == 1:
            date_test = row.date
            date_test = date_test.split(" ")
            date_t = date_test[0].strip()
            start_dates.append(date_t)
            
    boolean_features_dates["is a holiday"] = set(holiday_dates)
    boolean_features_dates["is a weekend"] = set(weekend_dates)
    boolean_features_dates["is start of a semester"] = set(start_dates)
    unique_dates = list(set(unique_dates))
    return unique_dates, boolean_features_dates

if __name__ == '__main__':
    
    gym_data, crowds_X_train, crowds_X_test, crowds_Y_train, crowds_Y_test = buildDataStructure()
    #unique_dates, unique_boolean_features_dates = testFeatures(gym_data)
    column_names =["Model","R Squared", "Mean Square Error","Root Mean Square Error", "Median Absolute Error","Max Error", "Explained Varience Score"]
    results_df = pd.DataFrame(columns = column_names)
   
    
# =============================================================================
#     #test linear regression
# =============================================================================
           
    lin_reg_model = LinearRegression()
    
    
    ## ravel() converts array to shape (n,) to avoid error message
    lin_reg_model.fit(crowds_X_train, crowds_Y_train.values.ravel())
    
    linear_regression_Y_pred = lin_reg_model.predict(crowds_X_test)
    linear_regression_Y_pred = linear_regression_Y_pred.astype(int)
    
    acc = accuracy_score(crowds_Y_test, linear_regression_Y_pred)
    
    #get coefficients and names into a dataframe
    coefficients = pd.DataFrame(zip(crowds_X_train.columns, np.transpose(lin_reg_model.coef_)), columns=['features', 'coef'])
    
    # =============================================================================
    #   print ("Linear regression co.efficients: \n",coefficients)
    #   print ("Linear regression intercepts: ",lin_reg_model.intercept_)
    #   print ("Linear regression accuracy score: ",acc)
    # =============================================================================
    
    
    results_df = results_df.append(getRegressionMetrics("linear regression",linear_regression_Y_pred), ignore_index=True)
    
    # try with feature eliminatiion
    
    
    lin_reg_model_rfe = LinearRegression()
    
    #Initializing RFE model
    rfe = RFECV(lin_reg_model_rfe)    
    # =============================================================================
    #     print("\nRecursive Feature elimination")
    #     print(rfe.support_)
    #     print(rfe.ranking_)
    # =============================================================================
    
    pipeline = Pipeline(steps=[('s',rfe),('m',lin_reg_model_rfe)])
    pipeline.fit(crowds_X_train,crowds_Y_train)
    
    linear_regression_Y_pred_rfe = pipeline.predict(crowds_X_test)
    linear_regression_Y_pred_rfe = linear_regression_Y_pred_rfe.astype(int)
    
    results_df = results_df.append(getRegressionMetrics("RFE linear regression",linear_regression_Y_pred_rfe), ignore_index=True)

# =============================================================================
#     # try decision tree regressor
# =============================================================================

    dec_tree_regressor = DecisionTreeRegressor(random_state = 0) 
  
    # fit the regressor with X and Y data
    dec_tree_regressor.fit(crowds_X_train, crowds_Y_train.values.ravel())
    
    dec_tree_crowds_Y_pred = dec_tree_regressor.predict(crowds_X_test)
    dec_tree_crowds_Y_pred = dec_tree_crowds_Y_pred.astype(int)
    
    results_df = results_df.append(getRegressionMetrics("Decision tree",dec_tree_crowds_Y_pred), ignore_index=True)
    
    
    # =============================================================================
    #     # export the decision tree to a tree.dot file
    #     # for visualizing the plot easily anywhere i.e http://www.webgraphviz.com/ 
    #     print(dec_tree_regressor.feature_importances_)
    #     features = crowds_X_train.columns.values
    #     export_graphviz(dec_tree_regressor, out_file ='tree.dot',
    #                    feature_names =features) 
    # =============================================================================
    
    #Initializing RFE model
    rfe = RFECV( DecisionTreeRegressor(random_state = 0) )    
    pipeline = Pipeline(steps=[('s',rfe),('m',DecisionTreeRegressor(random_state = 0))])
    pipeline.fit(crowds_X_train,crowds_Y_train)
    
    dec_tree_crowds_Y_pred_rfe = pipeline.predict(crowds_X_test)
    dec_tree_crowds_Y_pred_rfe = dec_tree_crowds_Y_pred_rfe.astype(int)
 
    results_df = results_df.append(getRegressionMetrics("RFE Decision tree",dec_tree_crowds_Y_pred_rfe), ignore_index=True)
    
# =============================================================================
#     try random forrest regressor
# =============================================================================

# =============================================================================
#     
    random_forest = RandomForestRegressor(n_estimators = 100, random_state = 42)
     
    random_forest.fit(crowds_X_train,crowds_Y_train.values.ravel())
    random_forest_predictions = random_forest.predict(crowds_X_test)
    random_forest_predictions = random_forest_predictions.astype(int)
    
    results_df = results_df.append(getRegressionMetrics("Random forest",random_forest_predictions), ignore_index=True)
#     
# =============================================================================
# =============================================================================
#     try Knn
# =============================================================================

    knn_regressor = KNeighborsRegressor(n_neighbors=12)
    # fit the model using the training data and training targets
    knn_regressor.fit(crowds_X_train,crowds_Y_train.values.ravel())
    
    knn_predictions = knn_regressor.predict(crowds_X_test)
    knn_predictions = knn_predictions.astype(int)
    results_df = results_df.append(getRegressionMetrics("K Nearest Neighbour",knn_predictions), ignore_index=True)
    
# =============================================================================
#     try polynomial regression
# =============================================================================