import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils._1_Imports.reqLibs import *
from utils._2_Cleaning.prepData import *
from utils._3_ModelTraining.trainModel import *

def evalute_MLP(mlp,xtest_scaled,ytest):
    #make prediciton with multilayer perceptron model and plot performance metrics
    #inputs mlp model and x-y test dataframes
   
    ypred = mlp.predict(xtest_scaled)
    print(confusion_matrix(ytest, ypred))
    print(accuracy_score(ytest, ypred))
    loss_values = mlp.loss_curve_
    # Plotting the loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(loss_values, label='Loss', color='blue')
    plt.title('Loss Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def evaluate_gridsearchestimations(grid):
    #print outputs of the grid search
    #wraps around grid search estimate to prints its results
   
    print(grid.best_params_)
    print(grid.best_score_)
    print(grid.estimator)
    
if __name__ == "__main__":
    # Test Code: Specify the path
    file_path = 'Dataset/Admission.csv'
    # Test Execution: Load the data and check it
    data = load_data(file_path)
    data = pre_prepdata(data)
    check_data(data)
    x,y,data,xtrain,xtest,ytrain,ytest,xtrain_scaled,xtest_scaled= cleanprep_and_split(data)
    mlp = train_MultiLayerPerceptron(data,xtrain_scaled,ytrain)
    evalute_MLP(mlp,xtest_scaled,ytest)
    grid = train_and_fit_using_GridSearchCV(mlp,x,y)
    evaluate_gridsearchestimations(grid)