# This is the main function that runs all the tasks. 1. Train and validate MLP. 2. Perform validation
from utils._1_Imports.reqLibs import *
from utils._2_Cleaning.prepData import *
from utils._3_ModelTraining.trainModel import *
from utils._4_ModelEvaluation.evalModel import *


file_path = 'CST2208_DataScience_TermProject\Predicting Chances of Admission at UCLA\Dataset\Admission.csv'
data = load_data(file_path)
data = pre_prepdata(data)
check_data(data) # Optional 
x,y,data,xtrain,xtest,ytrain,ytest,xtrain_scaled,xtest_scaled= cleanprep_and_split(data)
mlp = train_MultiLayerPerceptron(data,xtrain_scaled,ytrain)
evalute_MLP(mlp,xtest_scaled,ytest)
grid = train_and_fit_using_GridSearchCV(mlp,x,y)
evaluate_gridsearchestimations(grid)