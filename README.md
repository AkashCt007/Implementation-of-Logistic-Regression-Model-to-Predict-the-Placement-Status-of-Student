# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Load the dataset, drop unnecessary columns, and encode categorical variables.
2. Define the features (X) and target variable (y). 
3. Split the data into training and testing sets.
4. Train the logistic regression model, make predictions, and evaluate using accuracy and other

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: AKASH CT
RegisterNumber: 24901150 
*/
```
```
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv(r"C:\Users\admin\Desktop\Placement_Data.csv")
data.head()

datal = data.copy()
datal = datal.drop(["sl_no", "salary"], axis=1)
datal.head()

print("Missing values:\n", datal.isnull().sum())
print("Duplicate rows:", datal.duplicated().sum())

le = LabelEncoder()
datal["gender"] = le.fit_transform(datal["gender"])
datal["ssc_b"] = le.fit_transform(datal["ssc_b"])
datal["hsc_b"] = le.fit_transform(datal["hsc_b"])
datal["hsc_s"] = le.fit_transform(datal["hsc_s"])
datal["degree_t"] = le.fit_transform(datal["degree_t"])
datal["workex"] = le.fit_transform(datal["workex"])
datal["specialisation"] = le.fit_transform(datal["specialisation"])
datal["status"] = le.fit_transform(datal["status"])

x = datal.iloc[:, :-1]
y = datal["status"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

lr = LogisticRegression(solver="liblinear")
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

confusion = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", confusion)

classification_report_output = classification_report(y_test, y_pred)
print("Classification Report:\n", classification_report_output)
```

## Output:
## HEAD
![image](https://github.com/user-attachments/assets/435c7eb1-cc28-4938-819c-daad4906e4ef)
## COPY:
![image](https://github.com/user-attachments/assets/ba0f4d6f-3c31-480a-8877-3782932de6b2)
## FIT TRANSFORM:
![image](https://github.com/user-attachments/assets/6c1c5cca-84be-4770-bb27-217a80c0907a)
## LOGISTIC REGRESSION:
![image](https://github.com/user-attachments/assets/506c0385-98d6-4728-85c1-4f1991f76aa0)
## CONFUSION MATRIX:
![image](https://github.com/user-attachments/assets/1faec5b1-d29f-4dd7-b35d-f03a1feeddf9)
## ACCURACY SCORE:
![image](https://github.com/user-attachments/assets/c205ca6f-92c3-4f79-a4ab-6bc9baa4bb6d)
## CLASSIFICATION REPORT & PREDICITION:
![image](https://github.com/user-attachments/assets/11c3f08e-d8e1-4a14-b1d6-8fb747ab5700)






## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
