import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
iris=pd.read_csv('Iris.csv')
iris.head()
sns.pairplot(data=iris, hue='species', palette='Set2')
sns.pairplot(data=iris, hue='species', palette='Set2')
x=iris.iloc[:,:-1]
y=iris.iloc[:,4]
from sklearn.model_selection import train_test_split
x_train,x_test, y_train, y_test=train_test_split(x,y,test_size=0.30)
from sklearn.svm import SVC
model=SVC()
model.fit(x_train, y_train)
pred=model.predict(x_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,pred))
print(classification_report(y_test, pred))