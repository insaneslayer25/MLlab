import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
data = pd.read_csv('Iris.csv')
data['Species'] = data['Species'].replace({'Iris-setosa':1, 'Iris-versicolor':2 ,'Iris-virginica':3})
x = data[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y = data['Species']
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 0)
from sklearn.linear_model import LogisticRegression
reg = LogisticRegression(max_iter = 120)
reg.fit(x_train,y_train)
reg.predict(x_test)
print(reg.score(x_test,y_test))
y_pred = reg.predict(x_test)
print(y_pred)
from sklearn.metrics import confusion_matrix
cnf = confusion_matrix(y_test,y_pred)
print(cnf)
import seaborn as sns
plt.figure(figsize = (5,4))
sns.heatmap(cnf, annot=True)
plt.xlabel('actual')
plt.ylabel('predicted')
plt.show()

