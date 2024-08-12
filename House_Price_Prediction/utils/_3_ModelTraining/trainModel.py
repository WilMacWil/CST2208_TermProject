import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils._1_Imports.reqLibs import *

def train_linear_regression(x_train, y_train):
    #create linear regression model to be compared with decision tree and random forest models
    #inputs train x and y dataframes and returns model trained on data

    lrmodel = LinearRegression().fit(x_train, y_train)
    return lrmodel

def train_decision_tree(x_train, y_train):
    #create decision tree model to compare with linear regression and random forest
    #inputs x and y dataframes and then returns model trained on data

    dt = DecisionTreeRegressor(max_depth=3, max_features=10, random_state=567)
    dtmodel = dt.fit(x_train, y_train)
    return dtmodel

def train_random_forest(x_train, y_train):
    #create random forest model to be compared to linear regression and decision tree models
    #inputs train x and y dataframes and returns trained model
  
    rf = RandomForestRegressor(n_estimators=200, criterion='absolute_error')
    rfmodel = rf.fit(x_train, y_train)
    return rfmodel

def save_model(model, filename):
    #will be used to save the winning model
    #inputs the model to save and the name to save it under
    
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

def load_model(filename):
    #load model that was previously saved in a pickle file
    #inputs pickle file name and returns saved model
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model
