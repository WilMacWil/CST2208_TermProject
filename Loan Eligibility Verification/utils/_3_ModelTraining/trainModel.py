import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils._1_Imports.reqLibs import *
from utils._2_Cleaning.prepData import *

def train_LogisticRegression(xtrain_scaled,ytrain):
   #train logistical regression model to classify data
   #inputs x and y train dataframes and returns model trained on data
   
   
    lrmodel = LogisticRegression().fit(xtrain_scaled, ytrain)
    return lrmodel

def train_RandomForest(xtrain,ytrain):
    #train random forest classifier for comparison with logistics regression
    #inputs x and y train dataframes and returns model fit to data
   
    rfmodel = RandomForestClassifier(n_estimators=100, 
                                 min_samples_leaf=5, 
                                 max_features='sqrt')
    rfmodel.fit(xtrain, ytrain)
    return rfmodel 

if __name__ == "__main__":
    # Test Code: Specify the path
    file_path = 'Dataset/credit.csv'
                                                                                                                                                                   
    # Test Execution: Load the data and check it
    df = load_data(file_path)
    check_data(df)
    df = cleanandprep_data(df)
    xtrain,xtest,xtrain_scaled, xtest_scaled, ytrain, ytest = splitdata_and_Scale(df)
    lrmodel = train_LogisticRegression(xtrain_scaled, ytrain)
    rfmodel = train_RandomForest(xtrain, ytrain)
    if lrmodel:
        print(f" Logistic Regression Model: Successfully trained ")
        ypred = lrmodel.predict(xtest_scaled)
        print(ypred)
    if rfmodel:
        print(f" Random forest : Successfully trained ")
        ypred = rfmodel.predict(xtest)
        print(ypred)


   
    
        
        