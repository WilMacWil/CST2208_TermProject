from utils._1_Imports.reqLibs import *
from utils._2_Cleaning.prepData import *
from utils._3_ModelTraining.trainModel import *
from utils._4_ModelEvaluation.evalModel import *

#Mention the file path
file_path = 'CST2208_DataScience_TermProject\Loan Eligibility Verification\Dataset\credit.csv'

df = load_data(file_path)
# check_data(df) #OPTIONAL Uncomment if you wish to see some information
df = cleanandprep_data(df)
xtrain,xtest,xtrain_scaled, xtest_scaled, ytrain, ytest = splitdata_and_Scale(df)    
lrmodel = train_LogisticRegression(xtrain_scaled, ytrain)
rfmodel = train_RandomForest(xtrain, ytrain)
#Logistic regression model metrics
if lrmodel:
    print(f" Logistic Regression Model: Successfully trained ")
    evaluate_model_LogisticRegression(lrmodel,xtest_scaled,ytest)
#Random forest model metrics
if rfmodel:
    print(f" Random forest : Successfully trained ")
    evaluate_model_RandomForest(rfmodel,xtest,ytest)

kfolds_evaluations_lr(lrmodel,rfmodel,xtrain_scaled,ytrain)
