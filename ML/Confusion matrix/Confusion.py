from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
actual = [1,0,0,1,0,0,1,0,0,1]
predicted = [1,0,0,1,0,0,0,1,0,0]
matrix = confusion_matrix(actual,predicted, labels=[1,0])
print('Confusion matrix : \n',matrix)
tp, fn, fp, tn = confusion_matrix(actual,predicted,labels=[1,0]).reshape(-1)
print('Outcome values : \n', tp, fn, fp, tn)
matrix = classification_report(actual,predicted,labels=[1,0])
print('Classification report : \n',matrix)