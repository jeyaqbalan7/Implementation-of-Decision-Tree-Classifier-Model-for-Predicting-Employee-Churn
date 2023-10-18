# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the libraries and read the data frame using pandas.

2.Calculate the null values from dataframe and apply label encoder.

3.Apply decision tree classifier on the dataframe.

4.obtain the value of accuracy and data prediction. 

## Program:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Jeyabalan
RegisterNumber:  212222240040

import pandas as pd
data=pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
Initial dataset:

![image](https://github.com/SriramS22/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119094390/a89c4f08-8885-4cd3-aef9-3b184a40c0aa)

data info:

![image](https://github.com/SriramS22/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119094390/edab3131-9339-40a5-98d0-f81392e14ff3)

null values:

![image](https://github.com/SriramS22/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119094390/7467cc3f-1609-464f-a0e4-c463a31e19a4)

assignment of x and y values:

![image](https://github.com/SriramS22/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119094390/4f05e9f5-48af-445d-87a9-254cc415c1f5)

![image](https://github.com/SriramS22/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119094390/75f78e7d-f896-4385-a546-90f4408c5ced)

Converting string literals to numerical values using label encoder:

![image](https://github.com/SriramS22/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119094390/12469a7b-8436-460d-a816-02f6de0c544e)

Accuracy:

![image](https://github.com/SriramS22/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119094390/b2ed80e0-0374-416d-8827-e529a59b7c42)

Prediction:

![image](https://github.com/SriramS22/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119094390/43acb284-2d8a-4119-823c-8a9410dc3196)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
