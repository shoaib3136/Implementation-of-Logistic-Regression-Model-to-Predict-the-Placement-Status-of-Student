# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Import all necessary packages and dataset that you need to implement Logistic Regression.
2. Copy the actual dataset and remove fields which are unnecessary.
3. Then select dependent variable and independent variable from the dataset.
4. And perform Logistic Regression.
5. print the values of confusion matrix, accuracy, Classification report to find whether the student is placed or not. 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Shaik Shoaib Nawaz
RegisterNumber: 212222240094 
*/
import pandas as pd
import numpy as np
df=pd.read_csv('/content/Placement_Data.csv')
df
df1=df.copy()
df1
df1=df1.drop(['sl_no','salary'],axis=1)
df1.isnull().sum()
df1.duplicated().sum()
df1
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df1['gender']=le.fit_transform(df1['gender'])
df1['ssc_b']=le.fit_transform(df1['ssc_b'])
df1['hsc_b']=le.fit_transform(df1['hsc_b'])
df1['hsc_s']=le.fit_transform(df1['hsc_s'])
df1['degree_t']=le.fit_transform(df1['degree_t'])
df1['workex']=le.fit_transform(df1['workex'])
df1['specialisation']=le.fit_transform(df1['specialisation'])
df1['status']=le.fit_transform(df1['status'])
df1
x=df1.iloc[:,:-1]
x
y=df1['status']
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(solver="liblinear")
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy=accuracy_score(y_test,y_pred)
confusion=confusion_matrix(y_test,y_pred)
cr=classification_report(y_test,y_pred)
print("Accuracy Score:",accuracy)
print("\nConfusion Matrix:\n",confusion)
print("\nClassification Report:\n",cr)
from sklearn import metrics
cn_display=metrics.ConfusionMatrixDisplay(confusion_matrix=confusion,display_labels=['true','false'])
cn_display.plot()
```

## Output:
i.) Accuracy Score, Confusion Matrix and Classification Report:

![image](https://github.com/shoaib3136/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/117919362/3cfefc81-a2c3-4e29-8bfb-80c1292f6e3e)

ii.) Confusion Matrix:

![image](https://github.com/shoaib3136/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/117919362/98240d44-9cb0-45f3-8954-55b8ea78c155)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
