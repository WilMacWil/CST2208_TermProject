import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils._1_Imports.reqLibs import *
from utils._2_Cleaning.prepData import *

def decompisition_analysis(df):
    #idenitify any seasonal trend and remove from data
    #inputs the data cleaned in prep file and returns seasonally adjusted data
    
    """
     Decompose and plot AAPL trend and residual. This is a function to be used in conjunction with plot_harmonics
     
     @param df - dataframe with data to
    """
    decomposed = seasonal_decompose(df['AAPL'])
    trend = decomposed.trend
    seasonal = decomposed.seasonal
    residual = decomposed.resid
    plt.figure(figsize=(12,8))
    plt.subplot(411)
    plt.plot(df['AAPL'], label='Original', color='black')
    plt.legend(loc='upper left')
    plt.subplot(412)
    plt.plot(trend, label='Trend', color='red')
    plt.legend(loc='upper left')
    plt.subplot(413)
    plt.plot(seasonal, label='Seasonal', color='blue')
    plt.legend(loc='upper left')
    plt.subplot(414)
    plt.plot(residual, label='Residual', color='black')
    plt.legend(loc='upper left')
    plt.rcParams.update({'figure.figsize':(7,4), 'figure.dpi':80})
    # plot ACF
    plot_acf(df['AAPL'].dropna());
    # plot PACF
    plot_pacf(df['AAPL'].dropna(), lags=11);
    plt.show()

def train_arimamodel(df):
    #make wrapper for arima model
    #accepts trainning data as input which is then passed to arima model
    #returns trained arima model

    arima = ARIMA(df.AAPL, order=(1,1,1))
    ar_model = arima.fit()
    print(ar_model.summary())
    print("Arima model training complete")
    return ar_model

def train_arimamodel_bivariate(dfx):
    #similar to above but with additional columns
    #inputs dataframe to use for training with arimax
    #returns trained model

    model2 = ARIMA(dfx.AAPL, order=(1,1,1))
    arimax = model2.fit()
    print(arimax.summary())
    print("Arima Bivariate model training complete")
    print(dfx.head(5))
    return arimax

def train_XGBOOST(dataYF):
    #will use XGBoost as baseline model
    #input yahoo finance data 
    #returns trained models and predictions

    # Time series train test split chronologically (no random shuffling)
    train = dataYF.iloc[:-30]
    test = dataYF.iloc[-30:]
    # Be carefull not to use the next_day feature
    features = ['Open', 'High', 'Low', 'Close', 'Volume']

    model1 = XGBClassifier(max_depth=3, n_estimators=100, random_state=42)
    # Train the baseline model
    model1.fit(train[features], train['Target'])
    # Make predictions
    model1_preds = model1.predict(test[features])
    # Convert numpy array to pandas series
    model1_preds = pd.Series(model1_preds, index=test.index)
    print("Training XGBOOST Sucess")
    return train,test,model1,model1_preds,features
    
    
if __name__ == "__main__":
    # Test Code: Specify the path
    file_path = 'Dataset/AAPL.csv'
    # Test Execution: Load the data and check it
    data = load_data(file_path)
    df = pre_prep_data(data)
    check_data(df)
    decompisition_analysis(df)
    dfx = Bivariate_using_ExogenousVariable(data)
    train_arimamodel_bivariate(data)