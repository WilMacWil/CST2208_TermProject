import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils._1_Imports.reqLibs import *
from utils._2_Cleaning.prepData import *
from utils._3_ModelTraining.trainModel import *

def metrics_score(actual, predicted):
    #create confusion matrix for trained model
    #inputs y test and predictions and then plots confusion matrix

    print(classification_report(actual, predicted))
    cm = confusion_matrix(actual, predicted)
    plt.figure(figsize=(8,5))
    sns.heatmap(cm, annot=True,  fmt='.2f', xticklabels=['Not Attrite', 'Attrite'], yticklabels=['Not Attrite', 'Attrite'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    
def evaluation_LogisticRegression(lg,x_train,y_train,x_test,y_test):
    #calculate metrics of logistric regression baseline model
    #inputs the model and test-train x/y and outputs the accuracy score

    print("LR Accuracy on Train set: ")
    y_pred_train = lg.predict(x_train)
    metrics_score(y_train, y_pred_train)
    
    print("LR Accuracy on Test set: ")
    y_pred_test = lg.predict(x_test)
    metrics_score(y_test, y_pred_test)
    
def evaluation_SupportVectorMachine(svmmodel,x_train,y_train,x_test,y_test):
    #calcualte metrics for svm classifier
    #inputs the model and test-train x/y and then outputs the accuracy score

    print("SVM Accuracy using Linear Kernel on TRAIN set: ")
    y_pred_train_svm = svmmodel.predict(x_train)
    metrics_score(y_train, y_pred_train_svm)
    
    print("SVM Accuracy using Linear Kernel on TEST set: ")
    y_pred_test_svm = svmmodel.predict(x_test)
    metrics_score(y_test, y_pred_test_svm)
    
def evaluation_SVM_RBF(svmmodelRBF,x_train,y_train,x_test,y_test):
    #evaluates the svm with an RBF kernel and prints out metrics
    #inputs model and test-train x/y and then outputs the accuracy score
    
    print("SVM Accuracy using RBF Kernel on Train set: ")
    y_pred_train_svm = svmmodelRBF.predict(x_train)
    metrics_score(y_train, y_pred_train_svm)
    
    print("SVM Accuracy using RBF Kernel on Test set: ")
    y_pred_test_svm = svmmodelRBF.predict(x_test)
    metrics_score(y_test, y_pred_test_svm)
    
    
if __name__ == "__main__":
    # Test Code: Specify the path
    file_path = 'Dataset/HR_Employee_Attrition.xlsx'
    # Test Execution: Load the data and check it
    df = load_data(file_path)
    df,num_cols,cat_cols = pre_prep_data(df)
    analyze_numerical_and_categorical_columns(df,num_cols,cat_cols)
    df,X,Y,x_train,x_test,y_train,y_test,X_scaled = cleanprep_and_splitdata(df)
    lg = train_LogisticRegressionModel(x_train,y_train)
    evaluation_LogisticRegression(lg,x_train,y_train)
    svmmodel = train_SupportVectorMachines(x_train,y_train)
    evaluation_SupportVectorMachine(svmmodel,x_train,y_train,x_test,y_test)
    svmmodelRBF = train_SVM_RBF_KERNEL(x_train,y_train)
    evaluation_SVM_RBF(svmmodelRBF,x_train,x_test,y_test)
