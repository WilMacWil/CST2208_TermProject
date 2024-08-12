import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils._1_Imports.reqLibs import *

def load_data(file_path):
    #read data form csv and load into pandas dataframe
   
    data = pd.read_csv(file_path)
    return data



def pre_prepdata(data):
    #start by converting to binary category of admit yes/no and then remove unnecessary key column
    #input dataframe and returns dataframe with categorical column
  
    
    data['Admit_Chance']=(data['Admit_Chance'] >=0.8).astype(int)
    data = data.drop(['Serial_No'], axis=1)
    return data
    
def check_data(data): 
   #verify data from inconsistencies and plot it
   
    print(data.head())
    print(data.shape)
    print(data.info())
    print(data.describe().T)
    plt.figure(figsize=(15,8))
    sns.scatterplot(data=data, 
           x='GRE_Score', 
           y='TOEFL_Score', 
           hue='Admit_Chance');
    plt.title("Visualize the dataset to identify some patterns")
    plt.show()
    
def cleanprep_and_split(data):
    #scale data, convert categorical columns to dummies and do 80/20 train-test split
    #inputs data and returns scaled and split datasets
    
    data = pd.get_dummies(data, columns=['University_Rating','Research'])
    x = data.drop(['Admit_Chance'], axis=1)
    y = data['Admit_Chance']
    xtrain, xtest, ytrain, ytest =  train_test_split(x, y, test_size=0.2, random_state=123)
    # fit calculates the mean and standard deviation
    scaler = MinMaxScaler()
    scaler.fit(xtrain)
    # Now transform xtrain and xtest

    xtrain_scaled = scaler.transform(xtrain)
    xtest_scaled = scaler.transform(xtest)
    print("Data split successfully and transformed properly")
    return x,y,data,xtrain,xtest,ytrain,ytest,xtrain_scaled,xtest_scaled
    
if __name__ == "__main__":
    # Test Code: Specify the path
    file_path = 'CST2208_DataScience_TermProject\Predicting Chances of Admission at UCLA\Dataset\Admission.csv'
    # Test Execution: Load the data and check it
    data = load_data(file_path)
    data = pre_prepdata(data)
    check_data(data)
    data = cleanprep_and_split(data)