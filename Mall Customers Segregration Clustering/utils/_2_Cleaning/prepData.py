import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils._1_Imports.reqLibs import *

def load_data(file_path):
    #read data from csv file and load into pandas dataframe
    
    
    df = pd.read_csv(file_path)
    return df

def check_data(df): #OPTIONAL
    #verify data for any inconsistencies and make crossplots
    df.head()
    df.describe()
    df.shape
    sns.pairplot(df[['Age','Annual_Income','Spending_Score']])
    plt.show()
    
if __name__ == "__main__":
    # Test Code: Specify the path
    file_path = 'Dataset/mall_customers.csv'
    # Test Execution: Load the data and check it
    df = load_data(file_path)
    check_data(df)
   