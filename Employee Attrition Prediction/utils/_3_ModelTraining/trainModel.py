import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils._1_Imports.reqLibs import *
from utils._2_Cleaning.prepData import *


def train_LogisticRegressionModel(x_train,y_train):
    #train a logistic regression model as a baseline
    #inputs x and y train dataframes and returns model trained on data

    lg=LogisticRegression()
    lg.fit(x_train,y_train)
    print("Logistic Regression successfully trained")
    return lg
    
def train_SupportVectorMachines(x_train,y_train):
    #train the support vector model
    #inputs x and y train dataframes and returns model trained on data

    svm = SVC(kernel = 'linear') #linear kernal or linear decision boundary
    svmmodel = svm.fit(X = x_train, y = y_train)
    print("Support Vector machine successfully trained")
    return svmmodel

def train_SVM_RBF_KERNEL(x_train,y_train):
    #similar to svm above but using RBF kernal instead of linear
    #inputs x and y train dataframes and returns model trained on data

    svm = SVC(kernel = 'rbf') #RBF
    svmmodelRBF = svm.fit(X = x_train, y = y_train)
    print("SVM with RBF Kernel successfully trained")

    return svmmodelRBF

if __name__ == "__main__":
    # Test Code: Specify the path
    file_path = 'Dataset/HR_Employee_Attrition.xlsx'
    # Test Execution: Load the data and check it
    df = load_data(file_path)
    df,num_cols,cat_cols = pre_prep_data(df)
    analyze_numerical_and_categorical_columns(df,num_cols,cat_cols)
    df,X,Y,x_train,x_test,y_train,y_test,X_scaled = cleanprep_and_splitdata(df)
    train_LogisticRegressionModel(x_train,y_train)
    train_SupportVectorMachines(x_train,y_train)
    train_SVM_RBF_KERNEL(x_train,y_train)
