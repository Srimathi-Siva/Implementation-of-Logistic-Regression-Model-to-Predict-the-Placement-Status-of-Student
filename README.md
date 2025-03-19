# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages and print the present data.

2.Print the placement data and salary data.

3.Find the null and duplicate values.

4.Using logistic regression find the predicted values of accuracy , confusion matrices.

5.Display the results.
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SRI MATHI S
RegisterNumber:212224230272  
*/
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x
y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
TOP 5 ELEMENTS:

![ Image 2025-03-19 at 15 32 58_72df3eaa](https://github.com/user-attachments/assets/7e9afc7e-6c45-4224-986f-e5b249096347)

![ Image 2025-03-19 at 15 33 11_2ebaa42a](https://github.com/user-attachments/assets/24d4b30f-39de-4d45-a7a9-a53bc7ce69c3)

![ Image 2025-03-19 at 15 33 20_5397b3b7](https://github.com/user-attachments/assets/62bacd3e-7c9f-43f5-9251-f99deb5412dc)

DATA DUPLICATE:

![ Image4 2025-03-19 at 15 38 14_4f86f832](https://github.com/user-attachments/assets/1fb8b7bd-530d-491a-abaa-a96dff261e97)




PRINT DATA:

![ Image 2025-03-19 at 15 56 33_f9a3215b](https://github.com/user-attachments/assets/2ecb8de5-5afc-431e-93cc-015407085e1d)

Data-Status:

![image](https://github.com/user-attachments/assets/62d94f9c-9815-4667-a93c-f68d35a2a58b)

y_prediction array:

![image](https://github.com/user-attachments/assets/aeb0cd34-af17-4470-b1a1-65cf354bed02)

Confusion array:

![image](https://github.com/user-attachments/assets/9c6bc86b-921d-4732-99b3-597769946371)


Accuracy Value:

![image](https://github.com/user-attachments/assets/d35a8a52-4d53-43e5-8424-3667f165b315)


Classification Report:

![image](https://github.com/user-attachments/assets/7826253b-606b-453d-aaee-12fc17d37d5a)


Prediction of LR:

![image](https://github.com/user-attachments/assets/0510b061-8b24-4e29-bf4c-c11aa9318102)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
