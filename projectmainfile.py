#!/usr/bin/env python
# coding: utf-8

# In[71]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# In[72]:


df=pd.read_csv('SalaryData.csv')
df.head()


# In[73]:


df.shape


# In[74]:


df.info()


# In[75]:


df.columns


# In[76]:


df.describe()


# In[77]:


df.plot(x='TotalWorkingYears',y='MonthlyIncome',style='D')
plt.title('Working Years vs Monthly income')
plt.xlabel('TotalWorkingYears')
plt.ylabel('Salary')
plt.show()


# In[78]:


df.plot(x='Age',y='MonthlyIncome',style='D')
plt.title('Age vs Monthly income')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.show()


# In[79]:


df['JobRole'].value_counts().plot(kind='pie')


# In[80]:


df['Department'].value_counts().plot(kind='pie')


# In[81]:


sns.countplot(x='Gender', data=df)


# In[82]:


plt.figure(figsize=(10, 6))
sns.barplot(x='Department', y='MonthlyIncome', hue='Gender', data=df, errorbar =None)
plt.title('Monthly Income Comparison by Department and Gender')
plt.xlabel('Department')
plt.ylabel('Monthly Income')
plt.xticks(rotation=45, ha='right')
plt.show()


# In[83]:


plt.figure(figsize=(10, 6))
sns.barplot(x='JobRole', y='MonthlyIncome', hue='Gender', data=df, errorbar=None)
plt.title('Monthly Income Comparison by Department and Gender')
plt.xlabel('JobRole')
plt.ylabel('Monthly Income')
plt.xticks(rotation=45, ha='right')
plt.show()


# In[84]:


average_income_by_role = df.groupby('JobRole')['MonthlyIncome'].mean().reset_index()
colors = sns.color_palette("Set3", len(average_income_by_role))
plt.figure(figsize=(10, 6))
plt.bar(average_income_by_role['JobRole'], average_income_by_role['MonthlyIncome'])
plt.title('Average Monthly Income by Job Role')
plt.xlabel('Job Role')
plt.ylabel('Average Monthly Income')
plt.xticks(rotation=45, ha='right')
plt.show()



# In[85]:


average_income_by_department = df.groupby('Department')['MonthlyIncome'].mean().reset_index()


colors = sns.color_palette("Set2", len(average_income_by_department))


plt.figure(figsize=(10, 6))
plt.bar(average_income_by_department['Department'], average_income_by_department['MonthlyIncome'], color=colors)
plt.title('Average Monthly Income by Department')
plt.xlabel('Department')
plt.ylabel('Average Monthly Income')
plt.xticks(rotation=45, ha='right')

plt.show()



# In[86]:


average_percentage_hike_by_department = df.groupby('Department')['PercentSalaryHike'].mean().reset_index()


colors = sns.color_palette("Set2", len(average_percentage_hike_by_department))


plt.figure(figsize=(10, 6))
plt.bar(average_percentage_hike_by_department['Department'], average_percentage_hike_by_department['PercentSalaryHike'], color=colors)
plt.title('Average Percentage Hike by Department')
plt.xlabel('Department')
plt.ylabel('Average Percentage Hike')
plt.xticks(rotation=45, ha='right')

plt.show()


# In[87]:


average_working_years_by_department = df.groupby('Department')['TotalWorkingYears'].mean().reset_index()


average_working_years_by_department = average_working_years_by_department.sort_values(by='TotalWorkingYears', ascending=False)


plt.figure(figsize=(5, 6))
sns.barplot(x='TotalWorkingYears', y='Department', data=average_working_years_by_department, palette='viridis')
plt.title('Average Total Working Years by Department')
plt.xlabel('Average Total Working Years')
plt.ylabel('Department')

plt.show()



# In[88]:


average_income_by_gender = df.groupby('Gender')['MonthlyIncome'].mean().reset_index()
plt.figure(figsize=(5, 5))
sns.barplot(x='Gender', y='MonthlyIncome', data=average_income_by_gender, palette='pastel')
plt.title('Average Monthly Income by Gender')
plt.xlabel('Gender')
plt.ylabel('Average Monthly Income')
plt.show()



# In[89]:


plt.figure(figsize=(10, 6))
sns.countplot(x='JobRole', hue='Gender', data=df, palette='coolwarm')
plt.title('Distribution of Gender by Job Role')
plt.xlabel('Job Role')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')

plt.legend(title='Gender', loc='upper right', labels=['Male', 'Female'])

plt.show()



# In[90]:


df['Department'].unique()



# In[91]:


df['JobRole'].unique()


# In[92]:


df.isnull().count()


# In[93]:


df['Department']=df['Department'].map({'Sales':1,'Research & Development':2,'Human Resources':3})



# In[94]:


df['Gender']=df['Gender'].map({'Male':1,'Female':2})



# In[95]:


df['JobRole']=df['JobRole'].map({'Laboratory Technician':1,'Research Scientist':2,'Manufacturing Director':3,'Healthcare Representative':4,'Manager':5,'Sales Representative':6,'Research Director':7,'Sales Executive':8,'Human Resources':9})


# In[100]:


df.head()



# In[101]:


X=df.iloc[:,:-1]
y=df.iloc[:,-1]


# In[102]:


from sklearn.model_selection import train_test_split


# In[103]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.10, random_state=42)



# In[106]:


scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(X_train)
x_test_scaled = scaler.transform(X_test)
scaler


# In[107]:


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)


# In[117]:


y_pred=regressor.predict(X_test)



# In[109]:


df=pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
df




# In[110]:


from sklearn.metrics import r2_score
r2_score(y_test, y_pred)




# In[111]:


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

model = LinearRegression()
model.fit(X_train, y_train)
y_pred =model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")

rmse = np.sqrt(mse)

print(f'Root Mean Squared Error (RMSE): {rmse}')


mae = mean_absolute_error(y_test, y_pred)

print(f'Mean Absolute Error (MAE): {mae}')



# In[116]:


import scipy.stats as stats
residuals = y_test - y_pred

plt.figure(figsize=(8, 4))
stats.probplot(residuals, plot=plt, dist='norm', rvalue=True)

plt.xlabel('Theoretical')
plt.ylabel('Sample ')
plt.title('P-P Plot of Residuals')

plt.grid(True)
plt.show()






# In[113]:


correlation_matrix = x.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()




# In[122]:


from sklearn.model_selection import cross_val_score
model = LinearRegression()
cvscores = cross_val_score(model, X, y, cv=5, scoring='r2')
mean_r2 = cvscores.mean()
cvscores.mean().round(4)


# In[ ]:


import joblib

joblib.dump(regressor,'linear_regression_model.pkl')



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




