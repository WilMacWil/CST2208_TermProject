import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils._1_Imports.reqLibs import *
    
def load_data(file_path):
    #load data from csv into pandas dataframe

    df = pd.read_csv(file_path)
    return df

def split_data(df):
    #divide data into train-test with an 80-20 split
    x = df.drop('price', axis=1) 
    y = df['price']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test
