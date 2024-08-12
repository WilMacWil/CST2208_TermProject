import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils._1_Imports.reqLibs import *

def load_data(file_path):
   #laod data using pandas DataFrame
   
    df = pd.read_csv(file_path)
    return df

def prepare_and_splitdata(df):
    #scale and then 80-20 train-test split data 
    #inputs raw dataframe and returns scaled and split datasets
    
    Y= df.Attrition
    X= df.drop(columns = ['Attrition'])
    sc=StandardScaler()
    X_scaled=sc.fit_transform(X)
    X_scaled=pd.DataFrame(X_scaled, columns=X.columns)
    x_train,x_test,y_train,y_test=train_test_split(X_scaled,Y,test_size=0.2,random_state=1,stratify=Y)
    print("TRAIN TEST SPLIT SUCCESSFUL")
    return X,Y,X_scaled,x_train,x_test,y_train,y_test

if __name__ == "__main__":
    # Test Code: Specify the path
    file_path = 'Dataset/employee_attrition.csv'
    # Test Execution: Load the data and check it
    df = load_data(file_path)
    X,Y,X_scaled,x_train,x_test,y_train,y_test = prepare_and_splitdata(df)
