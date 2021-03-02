import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

#Importing the Data
df= pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/edx/project/drinks.csv')

#We use the method  head() to display the first 5 columns of the dataframe:
df.head()

#--------Question 1: Display the data types of each column using the attribute dtype.
df.dtypes

#--------Question 2 Use the method groupby to get the number of wine servings per continent:
df_gptest = df[['wine_servings','continent']]
grouped_test = df_gptest.groupby(['continent'],as_index=False).sum()
grouped_test.head()
#or ---------> df.groupby('continent')['wine_servings'].sum()

#--------Question 3: Perform a statistical summary and analysis of beer servings for each continent:
df_describe_beer_servings= df[['beer_servings', 'continent']]
df_describe_beer_servings.describe()
#or --------> df.groupby('continent')['beer_servings'].describe()

#-------Question 4: Use the function boxplot in the seaborn library to produce a plot that can be used to show the number of beer servings on each continent.
import seaborn as sns
df_gp_per_beer_servings = df[['beer_servings','continent']]
grouped_per_continent = df_gp_per_beer_servings.groupby(['continent'],as_index=False).sum()
sns.boxplot(x="beer_servings", y="continent", data=grouped_per_continent)
#or --------> df.boxplot(column = 'beer_servings', by='continent') --> plt.show()

#--------Question 5 Use the function regplot in the seaborn library to determine if the number of wine servings negatively or positively correlated with the number of beer servings.
import seaborn as sns
sns.regplot(x="beer_servings", y="wine_servings", data=df)
plt.ylim(0,)
df[["wine_servings", "beer_servings"]].corr()

#--------Question 6  Fit a linear regression model to predict the 'total_litres_of_pure_alcohol' using the number of 'wine_servings' then calculate  𝑅2
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
X = df[['wine_servings']] #predictor
Y = df['total_litres_of_pure_alcohol'] #target
lm.fit(X,Y)
print('The R-square is: ', lm.score(X, Y))
#to make a prediction
Yhat=lm.predict(X)
Yhat[0:5]
a = lm.intercept_
b = lm.coef_
#𝑌ℎ𝑎𝑡=𝑎+𝑏𝑋    ------>     total_litres_of_pure_alcohol = a + b x wine_servings
ex_prediction = a + b * 54 #54 wine servings
print('The prediction is', ex_prediction) # equal 4.86, it seems a good prediction
df[["wine_servings", "total_litres_of_pure_alcohol"]].corr()

#------Question 7: Use the list of features to predict the 'total_litres_of_pure_alcohol', split the data into training and testing and determine the R2 on the test data, using the provided code.
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
lre = LinearRegression()
y_data = df['total_litres_of_pure_alcohol']
x_data=df.drop('total_litres_of_pure_alcohol',axis=1)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.10, random_state=0)
lre.fit(x_train[['wine_servings']], y_train)
print("number of test samples :", x_test.shape[0])
print("number of training samples:",x_train.shape[0])
print('The R2 of the test data is is:',lre.score(x_test[['wine_servings']], y_test))
#or ------->
features =['beer_servings','spirit_servings','wine_servings']
X=df[features]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=0)
lm = LinearRegression()
lm.fit(X_train, y_train)
lm.score(X_test, y_test)

#--------Question8: Question 8 : Create a pipeline object that scales the data, performs a polynomial transform and fits a linear regression model. Fit the object using the training data in the question above, then calculate the R^2 using. the test data. Take a screenshot of your code and the  𝑅2. There are some hints in the notebook:
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]
pipe=Pipeline(Input)
pipe.fit(df[['wine_servings']],df['total_litres_of_pure_alcohol'])
#or ----------->
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)
#--------Question9: Create and fit a Ridge regression object using the training data, setting the regularization parameter to 0.1 and calculate the  𝑅2using the test data. Take a screenshot of your code and the  𝑅2
from sklearn.linear_model import Ridge
RR= Ridge(alpha=0.1)
RR.fit(X_train, y_train)
RR.score(X_test,y_test)

#-------Question10: Perform a 2nd order polynomial transform on both the training data and testing data. Create and fit a Ridge regression object using the training data, setting the regularization parameter to 0.1. Calculate the  𝑅2 utilizing the test data provided. Take a screen-shot of your code and the  𝑅2
from sklearn.linear_model import Ridge
pr=PolynomialFeatures(degree=2)
x_train_pr=pr.fit_transform(X_train)
x_test_pr=pr.fit_transform(X_test)
RR= Ridge(alpha=0.1)
RR.fit(x_train_pr, y_train)
RR.score(x_test_pr,y_test)
