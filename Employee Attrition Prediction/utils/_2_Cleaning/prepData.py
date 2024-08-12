import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils._1_Imports.reqLibs import *


def load_data(file_path):
    #load data from csv into pandas dataframe

    df = pd.read_excel(file_path)
    return df
def First_check_data(df): #Optional
    #verify data for any duplicates or nulls
    
    df.sample(5)
    df.info()
    df.nunique()
 
def pre_prep_data(df):
    #format data so its compatible with plotting
    #inputs dataframe and returns columns formatted for plotting
    
   
    df=df.drop(['EmployeeNumber','Over18','StandardHours'],axis=1)
    #Creating numerical columns
    num_cols=['DailyRate','Age','DistanceFromHome','MonthlyIncome','MonthlyRate','PercentSalaryHike','TotalWorkingYears',
            'YearsAtCompany','NumCompaniesWorked','HourlyRate',
            'YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager','TrainingTimesLastYear']

    #Creating categorical variables
    cat_cols= ['Attrition','OverTime','BusinessTravel', 'Department','Education', 'EducationField','JobSatisfaction','EnvironmentSatisfaction','WorkLifeBalance',
            'StockOptionLevel','Gender', 'PerformanceRating', 'JobInvolvement','JobLevel', 'JobRole', 'MaritalStatus','RelationshipSatisfaction']   
    print("Numerical and categorical columns succesfully created")
    return df,num_cols,cat_cols

    
def analyze_numerical_and_categorical_columns(df,num_cols,cat_cols):
    #plots all the cross charts and histograms for each coloumn
    #inputs dataframe, number and names of categorical columns and plots

    
    print(df[num_cols].describe().T)
    df[num_cols].hist(figsize=(14,14))
    plt.show()
    # Print the counts of all columns in the cat_cols
    for i in cat_cols:
        print(df[i].value_counts(normalize=True))
        print('*'*40)
     #OPTIONAL FOR FURTHER ANAYLSIS
    # for i in cat_cols:
    #     if i!='Attrition':
    #         (pd.crosstab(df[i],df['Attrition'],normalize='index')*100).plot(kind='bar',figsize=(8,4),stacked=True)
    #         plt.ylabel('Percentage Attrition %')
    print(df.groupby(['Attrition'])[num_cols].mean())
    plt.figure(figsize=(15,8))
    sns.heatmap(df[num_cols].corr(),annot=True, fmt='0.2f', cmap='YlGnBu')
    plt.show()

def cleanprep_and_splitdata(df):
    #prepare data for machine learning by converting categorical columns to numerical dummies and then scaling and test-train splitting
    #inputs dataframes and returns scaled and split data with dummy coloumns 


    #creating list of dummy columns
    to_get_dummies_for = ['BusinessTravel', 'Department','Education', 'EducationField','EnvironmentSatisfaction', 'Gender',  'JobInvolvement','JobLevel', 'JobRole', 'MaritalStatus' ]
    #creating dummy variables
    df = pd.get_dummies(data = df, columns= to_get_dummies_for, drop_first= True)
    #mapping overtime and attrition
    dict_OverTime = {'Yes': 1, 'No':0}
    dict_attrition = {'Yes': 1, 'No': 0}
    df['OverTime'] = df.OverTime.map(dict_OverTime)
    df['Attrition'] = df.Attrition.map(dict_attrition)
    #Separating target variable and other variables
    Y= df.Attrition
    X= df.drop(columns = ['Attrition'])
    sc=StandardScaler()
    X_scaled=sc.fit_transform(X)
    X_scaled=pd.DataFrame(X_scaled, columns=X.columns)
    x_train,x_test,y_train,y_test=train_test_split(X_scaled,Y,test_size=0.2,random_state=1,stratify=Y)
    print("Train test split completed")
    return df,X,Y,x_train,x_test,y_train,y_test,X_scaled

if __name__ == "__main__":
    # Test Code: Specify the path
    file_path = 'Dataset/HR_Employee_Attrition.xlsx'
    # Test Execution: Load the data and check it
    df = load_data(file_path)
    df,num_cols,cat_cols = pre_prep_data(df)
    analyse_numerical_and_categorical_columns(df,num_cols,cat_cols)
    df,X,Y,x_train,x_test,y_train,y_test,X_scaled = cleanprep_and_splitdata(df)
