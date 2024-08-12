from utils._1_Imports.reqLibs import *
from utils._2_Cleaning.prepData import *
from utils._3_ModelTraining.trainModel import *
from utils._4_ModelEvaluation.evalModel import *


# This is the main code that runs all the training and testing functions.
file_path = 'CST2208_DataScience_TermProject\Deep learning with Tensorflow\Dataset\employee_attrition.csv'
df = load_data(file_path)
X,Y,X_scaled,x_train,x_test,y_train,y_test = prepare_and_splitdata(df)
model_1 = train_Tensorflowmodel_run1(x_train,y_train)
model_1 = train_Tensorflowmodel_moreEpochs(model_1,x_train,y_train)
model_1 = train_Tensorflowmodel_MoreLayers(x_train, y_train)
model_1 = train_Tensorflowmodel_MoreNeurons_and_Morelayers(x_train,y_train)
model_1,history = activationfunctions_with_model(x_train,y_train)
check_best_learningrate(x_train,y_train)
predict_testset_using_activatedfunctions(model_1,x_test,y_test)
loss_curves_plot(history)