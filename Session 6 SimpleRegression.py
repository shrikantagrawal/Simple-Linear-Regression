import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Salary_Data.csv')  # Press F5 to set your working directory

# Split the dataset based in indendet ie Y and dependent ie X variable 
X=dataset.iloc[:,:-1].values  # [: - all rows, : -1 all columns minus 1]
Y=dataset.iloc[:,1].values     

# Splitting the complete dataset into training and test dataset
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=1/3,random_state=0)

# Implement classifier based on simple linear regression
from sklearn.linear_model import LinearRegression
SimpleLinearRegression = LinearRegression()
SimpleLinearRegression.fit(X_train,Y_train)

Y_predict= SimpleLinearRegression.predict(X_test)

# Implement the graph for simple linear regression
plt.scatter(X_train, Y_train, color='red')
plt.plot(X_train, SimpleLinearRegression.predict(X_train))
plt.show()
