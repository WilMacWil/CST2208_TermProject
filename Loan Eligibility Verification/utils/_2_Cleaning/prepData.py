import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils._1_Imports.reqLibs import *

#Load the data using the file path mentioned
def load_data(file_path):
    #read data from csv and load into pandas dataframe
    
    df = pd.read_csv(file_path)
    return df

#To Get Information on the Dataframe (Optional)
def check_data(df):
    #verify quality of data and if there are any nulls or wrong datatypes
    
    print(df.head(5))
    print(df.shape)
    df['Loan_Status'].value_counts().plot.bar()
    print(df.isnull().sum())
    print(df.dtypes)
    print(df['Dependents'].mode()[0])
    sns.displot(df['LoanAmount'])
    plt.show()
    
 
def cleanandprep_data(df):
   #handle any null values and get dummy values for categorical column
   
    df['Gender'].fillna('Male', inplace=True)
    df['Married'].fillna(df['Married'].mode()[0], inplace=True)
    df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
    df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)

    df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
    df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
    df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
    
    df = df.drop('Loan_ID', axis=1)
    df = pd.get_dummies(df, columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area'])
    #Debugging
    # print("Data types after get_dummies:")
    # print(df.dtypes)
    # print("Checking for null values after data cleaning:")
    # print(df.isnull().sum())
    
    return df


def splitdata_and_Scale(df):
    #prepare data for machine learning by scaling then splitting 80-20 for train-test
    #inputs dataframe and returns scaled and split data
    
    x = df.drop('Loan_Status',axis=1)
    y = df['Loan_Status']
    xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size=0.2, random_state=123)   
    # print(xtrain.shape, xtest.shape, ytrain.shape, ytest.shape)   #OPTIONAL TO CHECK THE SHAPE
    scale = MinMaxScaler()
    xtrain_scaled = scale.fit_transform(xtrain)
    xtest_scaled = scale.transform(xtest)
    return xtrain,xtest,xtrain_scaled, xtest_scaled, ytrain, ytest

    

if __name__ == "__main__":
    # Test Code: Specify the path
    file_path = 'Dataset/credit.csv'
    # Test Execution: Load the data and check it
    df = load_data(file_path)
    check_data(df)
    df = cleanandprep_data(df)
    xtrain_scaled, xtest_scaled, ytrain, ytest = splitdata_and_Scale(df)
    print(xtrain_scaled,xtest_scaled)
