# importing pandas
import pandas as pd
#importing numpy
import numpy as np
#importing matplotlib
import matplotlib.pyplot as plt
#importing seaborn
import seaborn as sb

df=pd.read_csv('concrete_data.csv')
                # study dataset
# print(df.head())
# checking for NULL values in data set and to remove
# print(df.info())
                # handling null values
# print(df.isnull().sum()) 
# dataset is clean no null values in columns.
# features are :-
# print(df.columns)
'''['cement', 'blast_furnace_slag', 'fly_ash', 'water', 'superplasticizer',
       'coarse_aggregate', 'fine_aggregate ', 'age',
       'concrete_compressive_strength']'

'''

# exploratory dataset analysis.
# pairplot of dataframe
sb.pairplot( df )
# plt.show()

# now scatter plotting for given dataset by pair wise to view relation.
# scatter plot of strength and cement 
plt.figure(figsize=[17,9])
plt.scatter(y='concrete_compressive_strength',x='cement',edgecolors='red',data=df)
plt.ylabel('concrete_compressive_strength')
plt.xlabel('cement')
# plt.show()

# scatter plot of flyash and strength
plt.figure(figsize=[17,9])
plt.scatter(y='csMPa',x='flyash',edgecolors='blue',data=df)
plt.ylabel('csMPa')
plt.xlabel('flyash')
# plt.show()

# now plotting corelation plot between features.
plt.figure(figsize=[17,8])

#ploting correlation plot

sb.heatmap(df.corr(),annot=True)
# plt.show()

# now performing box plot to get info about outlier in each feature.
'''
l=['cement','blast_furnace_slag','fly_ash','water','superplasticizer','coarse_aggregate','fine_aggregate ','age','concrete_compressive_strength']
for i in l:
  sb.boxplot(x=df[i])
  plt.show()
'''
# print(df.describe())
upper_limit=df.superplasticizer.mean()+3*df.superplasticizer.std()

df=df[df.superplasticizer<upper_limit]
# print(df.shape)
'''
plt.scatter(df['concrete_compressive_strength'],df['age'])
plt.show()
'''
# print(df['age'].describe())
upper_limit_age=df.age.mean()+3*df.age.std()

df=df[df.age <upper_limit_age]
# print(df.shape)
# print(df['age'].describe())
# print(df.describe())

# now spliting data set to indep and dep var
# independent variables
x = df.drop(['concrete_compressive_strength'],axis=1)
# dependent variables
y = df['concrete_compressive_strength']

# importing train_test_split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.3,random_state=42)
# now performing feature scaling on independent variable as data difference is very high prob of biasness.
from sklearn.preprocessing import StandardScaler
stand= StandardScaler()
Fit = stand.fit(x_train)
x_train_scl = Fit.transform(x_train)
x_test_scl = Fit.transform(x_test)
from sklearn.model_selection import StratifiedKFold
fold=StratifiedKFold(n_splits=10)


# finally applying model to our data set.
# 1. LINEAR REGRESSION
# import linear regression models

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
lr=LinearRegression()
fit=lr.fit(x_train_scl,y_train)
score = lr.score(x_test_scl,y_test)
print('predcted score is : {}'.format(score))
print('..................................')
y_predict = lr.predict(x_test_scl)
print('mean_sqrd_error is ==',mean_squared_error(y_test,y_predict))
rms = np.sqrt(mean_squared_error(y_test,y_predict)) 
print('root mean squared error is == {}'.format(rms))

'''
result of linear reg:-
predcted score is : 0.6146100224552373
..................................
mean_sqrd_error is == 93.38603996007897       
root mean squared error is == 9.66364527288119
'''
# Now, we plot a scatter plot and fit the line for checking the prediction values,

plt.figure(figsize=[17,8])
plt.scatter(y_predict,ytest)
plt.plot([ytest.min(), ytest.max()], [ytest.min(), ytest.max()], color='red')
plt.xlabel('predicted')
plt.ylabel('orignal')
plt.show()


# now applying new model 
# '''
# import rigd and lasso regresion
from sklearn.linear_model import Ridge,Lasso
from sklearn.metrics import mean_squared_error
rd= Ridge(alpha=0.4)
ls= Lasso(alpha=.56)
fit_rd=rd.fit(x_train_scl,y_train)
fit_ls = ls.fit(x_train_scl,y_train)
print('score od ridge regression is:-',rd.score(x_test_scl,y_test))
print('.......................................................')
print('score of lasso is:-',ls.score(x_test_scl,y_test))
print('mean_sqrd_roor of ridig is==',mean_squared_error(y_test,rd.predict(x_test_scl)))
print('mean_sqrd_roor of lasso is==',mean_squared_error(y_test,ls.predict(x_test_scl)))
print('root_mean_squared error of ridge is==',np.sqrt(mean_squared_error(y_test,rd.predict(x_test_scl))))
print('root_mean_squared error of lasso is==',np.sqrt(mean_squared_error(y_test,ls.predict(x_test_scl))))

# '''
'''
result of lass ridge
score od ridge regression is:- 0.6148072784500793      
.......................................................
score of lasso is:- 0.6239485628204591
mean_sqrd_roor of ridig is== 93.33824173674672
mean_sqrd_roor of lasso is== 91.12316506833604
root_mean_squared error of ridge is== 9.661171861464153
root_mean_squared error of lasso is== 9.54584543496992 
'''


plt.figure(figsize=[17,8])
plt.scatter(y_predict,ytest)
plt.plot([ytest.min(), ytest.max()], [ytest.min(), ytest.max()], color='red')
plt.xlabel('predicted')
plt.ylabel('orignal')
plt.show()




# ANOTHER MODEL

# import random forest regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
rnd= RandomForestRegressor(n_estimators=400,ccp_alpha=0.0)
fit_rnd= rnd.fit(x_train_scl,y_train)
print('score is:-',rnd.score(x_test_scl,y_test))
print('........................................')
print('mean_sqrd_error is==',mean_squared_error(y_test,rnd.predict(x_test_scl)))
print('root_mean_squared error of is==',np.sqrt(mean_squared_error(y_test,rnd.predict(x_test_scl))))

# result
# score is:- 0.8695475561000607 ,.87 with 400 n estimators
# ........................................
# mean_sqrd_error is== 31.610674508302978 
# root_mean_squared error of is== 5.622337103758808



from sklearn.metrics import r2_score
x_predict = list(rnd.predict(x_test))
predicted_df = {'predicted_values': x_predict, 'original_values': y_test}
#creating new dataframe
result=pd.DataFrame(predicted_df)
# print(result.head(30))

# import pickle
# file = 'concrete_strength'
# save = pickle.dump(rnd,open(file,'wb'))
