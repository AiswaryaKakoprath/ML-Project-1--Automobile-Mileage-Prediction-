#!/usr/bin/env python
# coding: utf-8

# # Objective of Automobile mileage prediction project:
# Build a predictive modeling algorithm to predict mileage of cars based on given input variables

# In[1]:


#import libraries
#-------------------
#for data preparation and analysis
import pandas as pd
#for creating plot
import matplotlib.pyplot as plt
#for distribution plot and heatmap
import seaborn as sns

#for craeting training and test samples
from sklearn.model_selection import train_test_split

#feature selection(to select significant variables)
from sklearn.feature_selection import SelectKBest, f_regression

#for building linear regression model
from sklearn.linear_model import LinearRegression


# In[2]:


df=pd.read_csv(r"C:\Users\aksha\OneDrive\Desktop\Introtallent\python\Data Files used in Projects\automobile data.csv")


# In[3]:


df.head()


# In[4]:


df.columns


# # Variable Description
# * Target Variable(y):MPG(Mileage per gallon)
# * Independent variable(x):
#     * Cylinders'
#     * 'Displacement'
#     * 'Horsepower'
#     * 'Weight'
#     * Acceleration'
#     * 'Model_year'
#     * 'Origin'
#     * 'Car_Name']

# In[5]:


df.tail()


# In[6]:


df.shape
#observation: 398
#variables: 9


# In[7]:


#check the data type
df.dtypes
#categorical variables are must be stored as numeric


# In[8]:


#horsepower is numeric variable but in the df it is stored as categorical
#so we need to change the data type of horsepower
#it will decide whether it is an integer or float
df['Horsepower']=pd.to_numeric(df['Horsepower'], errors='coerce')
# errors='coerce', it means that if there's an error encountered during the conversion or parsing process, 
# the problematic values will be set to NaN (Not a Number) instead of raising an error or stopping the operation entirely.


# In[9]:


#check the data type again
df.dtypes


# In[10]:


#descriptive statistic
df.describe()


# In[11]:


#check missing values in the data
df.isnull().sum()


# In[12]:


#there are 6 missing values in horsepower variable
#missing value imputation
df['Horsepower']=df['Horsepower'].fillna(df['Horsepower'].median())


# In[13]:


df.isnull().sum()


# In[14]:


#check outlier in  numeric column
plt.boxplot(df['MPG']) #no outlier
plt.show()


# In[15]:


plt.boxplot(df['Cylinders']) #no outlier
plt.show()


# In[16]:


plt.boxplot(df['Displacement']) #no outlier
plt.show()


# In[17]:


plt.boxplot(df['Horsepower']) #there is outlier
plt.show()


# In[18]:


plt.boxplot(df['Acceleration']) #there is outlier
plt.show()


# In[19]:


plt.boxplot(df['Origin']) #no outlier
plt.show()


# In[20]:


def remove_outliers(d,c):
    #where d is the variable for dataframe and c is the variable for column
    
    #find q1 and q3
    q1=d[c].quantile(0.25)
    q3=d[c].quantile(0.75)
    
    #calculate iqr(inter quartile range)
    iqr=q3-q1
    
    #find upper bound (ub) and lower bound(lb)
    ub=q3+1.5*iqr
    lb=q1-1.5*iqr
    
    #filter goood data(ie eliminate outliers)
    data_without_outlier=d[(d[c]<=ub) & (d[c]>=lb)]
    
    return data_without_outlier


# In[23]:


#remove outliers from horsepower variable
df=remove_outliers(df,'Horsepower')

plt.boxplot(df['Horsepower'])
plt.show()


# In[26]:


#remove outliers from acceleration variable
df=remove_outliers(df,'Acceleration')

plt.boxplot(df['Acceleration'])
plt.show()


# # EDA (Exploratory Data Analysis)
# * Distribution
# * Data mix
# * correlation

# In[27]:


df.columns #checking the data using eda


# In[28]:


df.shape


# In[29]:


#check skewness(distribution)
sns.distplot(df['MPG'])


# In[40]:


sns.distplot(df['Displacement'])


# In[30]:


sns.distplot(df['Horsepower'])


# In[31]:


sns.distplot(df['Weight'])


# In[43]:


sns.distplot(df['Acceleration'])


# In[44]:


#check data mix for categorical variable
#cylinder,model_year,origin,car_name


# In[32]:


df.groupby('Cylinders')['Cylinders'].count().plot(kind='bar')


# In[33]:


df.groupby('Model_year')['Model_year'].count().plot(kind='bar')


# In[34]:


df.groupby('Origin')['Origin'].count().plot(kind='bar')


# In[35]:


df.groupby('Car_Name')['Car_Name'].count().plot(kind='bar')


# # Pearson correlation

# In[36]:


#create a set of numeric columns
df_numeric=df.select_dtypes(include=['int64','float64'])
df_numeric.head()


# In[37]:


#n df_numeric weve categorical variable. we neded to drop that
df_numeric=df_numeric.drop(['Cylinders','Model_year','Origin'],axis=1)
df_numeric.head()


# In[38]:


#create heat map
sns.heatmap(df_numeric.corr(),cmap='YlGnBu',annot=True)


# In[ ]:


#using the pearson correlation test we found thet key drivers(input variables) 


# --------------------End of EDA----------------------------

# In[ ]:


#check if there is any problem in categorical variables.
#like spelling difference, case sensitive value ie,Male, male


# In[39]:


df.columns


# In[40]:


df['Cylinders'].unique()


# In[41]:


df['Model_year'].unique()


# In[42]:


df['Origin'].unique()
#origin 1 US, 2 Germany, 3 Japan


# In[43]:


df['Car_Name'].unique()


# # Dummy Conversion(One-hot encoding)

# In[44]:


#remove model year as it doesnt signify anything in terms of impact on target variable
df=df.drop('Model_year',axis=1)


# In[45]:


df.dtypes


# In[46]:


#cylinders and origin are categorical variables stored as numeric.
#hence we need to change the datatype of these variables to object
df['Cylinders']=df['Cylinders'].astype('object')
df['Origin']=df['Origin'].astype('object')
df.dtypes


# In[47]:


#create a new df to store categorical variable for dummy conversion
df_categorical=df.select_dtypes(include='object')
df_categorical.head()


# In[48]:


#dummy conversion
df_dummy=pd.get_dummies(df_categorical,drop_first=True)
df_dummy.head()


# In[49]:


#combine data from df_numeric and df_dummy
df_final=pd.concat([df_numeric,df_dummy], axis=1)
df_final.head()


# In[50]:


#create x and y

x=df_final.drop('MPG',axis=1)

y=df_final['MPG']


# In[ ]:


#from sklearn.feature_selection import RFE
#from sklearn.linear_model import LinearRegression


# In[52]:


#testing and test sample
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=999)


# In[53]:


#check sample size
print(xtrain.shape,xtest.shape,ytrain.shape,ytest.shape)


# # FEATURE SELECTION
# Select significant variables

# In[ ]:


#p value to test the significance of the hypothetical mechanism. p value of variable <0.05 significant,confidence is 95%
#VIF factor tells you to how expencive(non relevent) that variable,how useless is , we choose small vif.


# In[ ]:


#VIF (variable inflation factor) it is a score that tells as the relevency of a variable.
#along with p value we also check VIF to find the significant variable.
#VIF= 1/(1-R^2)
#P value: measures the strength of evidence against null hypothesis.
#a variable with p value < 0.05 is considered as significant variable.
#while finding the significant variables manualy we  should also check VIF score


# In[60]:


#create a key_features object to select the top k features
# key_features=selectKBest(score_func=f_regression,k='all')

key_features=SelectKBest(score_func=f_regression,k=5) #to select 5  significant features

#fit the key featuresto the training data and transform it
xtrain_selected=key_features.fit_transform(xtrain,ytrain)

#get the indices of the selected features
selected_indices=key_features.get_support(indices=True)

#get the name of the selected features
selected_features=xtrain.columns[selected_indices]


# In[61]:


#print the significant variables
selected_features


# # Build Linear Regression Model

# In[62]:


#instantiate the linear regression function
linreg=LinearRegression()

#fit the model using training sample
linreg.fit(xtrain_selected,ytrain)

#print the accuracy of training model
linreg.score(xtrain_selected,ytrain)


# In[63]:


#evaluate the model on the test set using the selected features
xtest_selected=xtest.iloc[:,selected_indices]
score=linreg.score(xtest_selected,ytest)
score


# In[65]:


#predict mileage based on xtest
predicted_mpg=linreg.predict(xtest_selected)

#check the accuracy of test model
linreg.score(xtest_selected,ytest)


# In[66]:


#print predicted mileage
predicted_mpg


# In[67]:


#print B0
linreg.intercept_


# In[68]:


#print beta values
linreg.coef_


# In[ ]:




