# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Use the standard libraries in python for finding linear regression.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Predict the values of array.
5. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
6. Obtain the graph.

## Program:
```python
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: JAYABHARATHI S
RegisterNumber:  212222100013
*/
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

df=np.loadtxt("/content/ex2data1.txt",delimiter=',')
X=df[:,[0,1]]
y=df[:,2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.show()

def sigmoid(z):
  return 1/(1+np.exp(-z))

plt.plot()
X_plot = np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costFunction(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  J=-(np.dot(y,np.log(h)) + np.dot(1-y,np.log(1-h))) / X.shape[0]
  grad=np.dot(X.T,h-y)/X.shape[0]
  return J,grad

X_train = np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

X_train = np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

def cost(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h))) / X.shape[0]
  return J

def gradient(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  grad=np.dot(X.T,h-y) / X.shape[0]
  return grad

X_train = np.hstack((np.ones((X.shape[0],1)),X))
theta = np.array([0,0,0])
res = optimize.minimize(fun=cost,x0=theta,args=(X_train,y),
                        method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
  x_min,x_max=X[:,0].min() - 1,X[:,0].max()+1
  y_min,y_max=X[:,1].min() - 1,X[:,0].max()+1
  xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),
                    np.arange(y_min,y_max,0.1))

  X_plot = np.c_[xx.ravel(),yy.ravel()]
  X_plot = np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
  y_plot = np.dot(X_plot,theta).reshape(xx.shape)

  plt.figure()
  plt.scatter(X[y==1][:,0],X[y==1][:,1],label='admitted')
  plt.scatter(X[y==0][:,0],X[y==0][:,1],label='NOT admitted')
  plt.contour(xx,yy,y_plot,levels=[0])
  plt.xlabel("Exam 1 score")
  plt.ylabel("Exam 2 score")
  plt.legend()
  plt.show()

plotDecisionBoundary(res.x,X,y)

prob = sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,X):
  X_train = np.hstack((np.ones((X.shape[0],1)),X))
  prob = sigmoid(np.dot(X_train,theta))
  return (prob>=0.5).astype(int)

np.mean(predict(res.x,X)==y)

```

## Output:

## Array Value of x:

![image](https://github.com/Jayabharathi3/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/120367796/a13aecec-586d-4edf-ae20-b6294e49cc3b)

## Array Value of y :
![image](https://github.com/Jayabharathi3/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/120367796/77c1cee2-5e0e-464a-bc17-0cc2f258925d)

## Exam 1 - score graph :
![image](https://github.com/Jayabharathi3/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/120367796/d3eb2e25-d6b7-4383-b46e-ddd055c12fdc)

## Sigmoid function graph :
![image](https://github.com/Jayabharathi3/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/120367796/26309aad-9f4b-4e96-8071-3bef9938e2af)

## X_train_grad value :
![image](https://github.com/Jayabharathi3/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/120367796/7e80a000-4ab2-4000-ae45-18fb78491b82)

## Y_train_grad value :
![image](https://github.com/Jayabharathi3/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/120367796/a02f79d9-4264-4a4a-b48a-d5cdf37a5195)

## Print res.x :
![image](https://github.com/Jayabharathi3/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/120367796/ca2bb7b1-d31f-4fc1-abe2-55876dc131ea)

## Decision boundary - graph for exam score :
![image](https://github.com/Jayabharathi3/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/120367796/c188ff15-0815-4ee3-b5cf-2248a9f4c545)

## Proability value :
![image](https://github.com/Jayabharathi3/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/120367796/4ab73e9a-f5d1-4e66-a7c5-f314d959816b)

## Prediction value of mean :
![image](https://github.com/Jayabharathi3/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/120367796/fa569780-dab4-4fb4-8c51-1a821b40c3c2)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

