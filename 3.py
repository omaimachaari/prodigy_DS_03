import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import tree
import matplotlib.pyplot as plt


df= pd.read_csv(r"c:\Users\lenovo\Desktop\bank-full.csv", delimiter=';')

print('checking for empty cells number=',df.isnull().sum().sum()) 

df=pd.DataFrame(df)
df.rename(columns={'poutcome': 'Previous_mkt_outcome', 'p': 'client_signed_deposit'}, inplace=True)

X = df.drop(columns='y')
y = df['y']
print(df.dtypes)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
dt_classifier = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
dt_classifier.fit(X_train, y_train)

y_pred = dt_classifier.predict(X_test)
