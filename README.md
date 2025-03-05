# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary libraries including NumPy, Pandas, and StandardScaler from sklearn.preprocessing.
2.Define linear_regression function. Begin by reading the dataset using the read_csv function from Pandas. 
3. Apply the linear_regression function to the standardized input features X1_Scaled and target variable Y1_Scaled to obtain the optimal parameters theta
4. Prepare new data and make predictions using the trained model.
5. Print out the predicted value obtained from the regression analysis.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: NAVEEN.S
RegisterNumber: 212223240106 
*/
```
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1, y, learning_rate=0.01, num_iters=1000):
    X = np.c_[np.ones(len(X1)), X1]
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1,1)
        errors = (predictions - y).reshape(-1,1)
        theta -= learning_rate * (1 / len(X1)) * X.T.dot(errors)
    return theta
data = pd.read_csv('50_Startups.csv',header=None)
X = (data.iloc[1:, :-2].values)
X1=X.astype(float)
scaler = StandardScaler()
y = (data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled = scaler.fit_transform(X1)
Y1_Scaled = scaler.fit_transform(y)
theta = linear_regression(X1_Scaled, Y1_Scaled)

new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted Value: {pre}")
```

## Output:
![image](https://github.com/user-attachments/assets/164c0bdc-23f5-4510-98a5-a9ae065327f3)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
