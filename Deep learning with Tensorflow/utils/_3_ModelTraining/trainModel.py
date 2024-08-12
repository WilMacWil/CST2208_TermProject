import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils._1_Imports.reqLibs import *
from utils._2_Cleaning.prepData import *

def train_Tensorflowmodel_run1(x_train, y_train):
    #train tensor flow model built with keras sequential layers
    #inputs the scaled train x and y dataframes and returns model trained on data
    

    # set a fixed random seed for the model's weight initialization
    tf.keras.utils.set_random_seed(42)

    # 1. Create the model using the Sequential API
    model_1 = tf.keras.Sequential([
    tf.keras.layers.Dense(1) #output layer
    ])

    # 2. Compile the model
    model_1.compile(loss=tf.keras.losses.BinaryCrossentropy(), # binary since we are working with 2 clases (0 & 1)
                    optimizer=tf.keras.optimizers.SGD(),
                    metrics=['accuracy'])

    # 3. Fit the model
    model_1.fit(x_train, y_train, epochs=5,verbose=0)
    print("Tensorflow First evaluation using Sequential API:")
    model_1.evaluate(x_train, y_train)
    return model_1

def train_Tensorflowmodel_moreEpochs(model_1,x_train,y_train):
    #similar to function above but trained with more epochs
    #input is basically everything from the function above and returns trained model
   
    
    tf.keras.utils.set_random_seed(42)
    print("Tensorflow Second evaluation using more Epochs:")
    # Train our model for longer (more chances to look at the data)
    history = model_1.fit(x_train, y_train, epochs=100, verbose=0) # set verbose=0 to remove training updates
    model_1.evaluate(x_train, y_train)
    return model_1

def train_Tensorflowmodel_MoreLayers(x_train, y_train):
    #take model from first function and add another layer to it
    #input is x and y train dataframes and returns trained model

    

    tf.keras.utils.set_random_seed(42)
    # set model_1 to None
    model_1 = None

    # 1. Create the model (same as model_1 but with an extra layer)
    model_1 = tf.keras.Sequential([
    tf.keras.layers.Dense(1), # add an extra layer
    tf.keras.layers.Dense(1) # output layer
    ])

    # 2. Compile the model
    model_1.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                    optimizer=tf.keras.optimizers.SGD(),
                    metrics=['accuracy'])

    # 3. Fit the model
    model_1.fit(x_train, y_train, epochs=50,verbose=0)
    print("Tensorflow Third evaluation using more Layers:")
    model_1.evaluate(x_train, y_train)
    return model_1

def train_Tensorflowmodel_MoreNeurons_and_Morelayers(x_train,y_train):
    #same model but with additional neurons and layers
    #inputs x and y train, returns model fit to data
    
  
    # set a fixed random seed for the model's weight initialization
    tf.keras.utils.set_random_seed(42)

    # set model_1 to None
    model_1 = None

    # 1. Create the model (same as model_1 but with an extra layer)
    model_1 = tf.keras.Sequential([
    tf.keras.layers.Dense(1),
    tf.keras.layers.Dense(1), # add another layer with 1 neuron
    tf.keras.layers.Dense(1) # output layer
    ])

    # 2. Compile the model
    model_1.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                    optimizer=tf.keras.optimizers.SGD(),
                    metrics=['accuracy'])

    # 3. Fit the model
    model_1.fit(x_train, y_train, epochs=50,verbose=0)
    print("Tensorflow Fourth evaluation using more Layers and Neurons:")
    model_1.evaluate(x_train, y_train)
    return model_1

def activationfunctions_with_model(x_train,y_train):
    #this time the 2 layers have a different number of neurons
    #inputs x and y train dataframes and returns model fit to data

    # set a fixed random seed for the model's weight initialization
    tf.keras.utils.set_random_seed(42)

    # set model_1 to None
    model_1 = None

    # 1. Create the model (same as model_1 but with an extra layer)
    model_1 = tf.keras.Sequential([
    tf.keras.layers.Dense(1), # try activations LeakyReLU, sigmoid, Relu, tanh. Default is Linear
    tf.keras.layers.Dense(1, activation = 'sigmoid') # output layer
    ])

    # 2. Compile the model
    model_1.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                    optimizer=tf.keras.optimizers.SGD(learning_rate=0.0009),
                    metrics=['accuracy'])

    # 3. Fit the model
    history = model_1.fit(x_train, y_train, epochs=50,verbose=0)
    print("Tensorflow Fifth evaluation using Activation Function")
    model_1.evaluate(x_train, y_train)
    return model_1,history


if __name__ == "__main__":
    # Test Code: Specify the path
    file_path = 'Dataset/employee_attrition.csv'
    # Test Execution: Load the data and check it
    df = load_data(file_path)
    X,Y,X_scaled,x_train,x_test,y_train,y_test = prepare_and_splitdata(df)
    model_1 = train_Tensorflowmodel_run1(x_train,y_train)
    train_Tensorflowmodel_moreEpochs(model_1,x_train,y_train)
    train_Tensorflowmodel_MoreLayers(x_train, y_train)
    train_Tensorflowmodel_MoreNeurons_and_Morelayers(x_train,y_train)
    activationfunctions_with_model(x_train,y_train)
    
