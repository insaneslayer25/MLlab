import pandas as pd
data=pd.read_csv("Heart.csv")
X =data.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12]].values
y =data.iloc[:,13].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred=gnb.predict(X_test)
from sklearn.metrics import accuracy_score,confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)
print("Classification Accuracy:", accuracy_score(y_test,y_pred))
print("Classification Accuracy:", 1-accuracy_score(y_test,y_pred))
import seaborn as sn
from matplotlib import pyplot as plt
sn.heatmap(cm,annot=True)
plt.xlabel('predicted value')
plt.ylabel('Truth or Actual value')
plt.show()