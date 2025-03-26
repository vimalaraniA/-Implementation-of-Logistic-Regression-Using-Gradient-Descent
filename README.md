# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

 1.Import Necessary Libraries:
Import NumPy, pandas, and StandardScaler for numerical operations, data handling, and feature scaling, respectively.

2.Define the Linear Regression Function:
Create a linear regression function using gradient descent to iteratively update parameters, minimizing the difference between predicted and actual values.

3.Load and Preprocess the Data:
Load the dataset, extract features and target variable, and standardize both using StandardScaler for consistent model training.

4.Perform Linear Regression:
Apply the defined linear regression function to the scaled features and target variable, obtaining optimal parameters for the model.

5.Make Predictions on New Data:
Prepare new data, scale it, and use the trained model to predict the target variable, transforming predictions back to the original scale.

 6.Print the Predicted Value

<br>
<br>
<br>

## Program:

```python
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: VIMALA RANI A
RegisterNumber:  212223040240
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv("Placement_Data.csv")
dataset

#dropping the serial no and salary col
dataset=dataset.drop("sl_no",axis=1)
dataset=dataset.drop("salary",axis=1)

dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes

#labelling the columns
dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset

#selecting the features and labels
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
#display independent variable
Y

#initialize the model parameters
theta=np.random.randn(X.shape[1])
y=Y
#define the sigmoid function
def sigmoid(z):
    return 1/(1+np.exp(-z))
#define the loss function
def loss(theta,X,y):
    h=sigmoid(X.dot(theta))
    return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))

#defining the gradient descent algorithm.
def gradient_descent(theta,X,y,alpha,num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h-y)/m
        theta -= alpha*gradient
    return theta
#train the model
theta = gradient_descent(theta,X,y,alpha=0.01,num_iterations=1000)
#makeprev \dictions
def predict(theta,X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h>=0.5,1,0)
    return y_pred
y_pred = predict(theta,X)


accuracy=np.mean(y_pred.flatten()==y)
print("Accuracy:",accuracy)
print(Y)

xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)

xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
```



## Output:


### Read the file and display
![Screenshot 2025-03-26 212233](https://github.com/user-attachments/assets/1b562f15-22a2-4b46-b8bb-1cae5605fb31)


### Categorizing columns

![Screenshot 2025-03-26 212245](https://github.com/user-attachments/assets/b713cd5a-a550-4fdd-86d3-54b42970b341)


### Labelling columns and displaying dataset

![Screenshot 2025-03-26 212258](https://github.com/user-attachments/assets/553e59a7-34a7-479b-8784-95fe915472eb)


### Display dependent variable
![Screenshot 2025-03-26 212310](https://github.com/user-attachments/assets/f0b1ddf9-5ae0-4115-b96b-a169f1b28ccd)


### Printing accuracy
![Screenshot 2025-03-26 212320](https://github.com/user-attachments/assets/9e79ce64-0b30-4ffc-bbd2-86558d0d306e)


### Printing Y
![Screenshot 2025-03-26 212329](https://github.com/user-attachments/assets/a1d2eeac-f8c4-44f7-8f2f-a7ae9ba6f75c)


### Printing y_prednew

![Screenshot 2025-03-26 212336](https://github.com/user-attachments/assets/b41be7f6-aa7b-4eeb-8df9-4fe3308b024e)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

