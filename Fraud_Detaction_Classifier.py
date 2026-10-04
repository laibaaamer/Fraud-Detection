import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix,classification_report,ConfusionMatrixDisplay,RocCurveDisplay
df=pd.read_csv(r"C:\Users\Laiba Aamer\Downloads\creditcard.csv")
print("Data Shape: ",df.shape)
x=df.drop("Class",axis=1)
y=df["Class"]
x_train,x_test,y_train,y_test = train_test_split(
    x,y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
model=LogisticRegression(max_iter=1000)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
y_prob=model.predict_proba(x_test)[:,1]
print("Classification Metrics")
print("Accuracy: ",accuracy_score(y_test,y_pred))
print("Precision: ",precision_score(y_test,y_pred))
print("Recall: ",recall_score(y_test,y_pred))
print("F1: ",f1_score(y_test,y_pred))
print("ROC-AUC: ",roc_auc_score(y_test,y_prob))
print("Classification Report")
print(classification_report(y_test,y_pred))
print("Confusion Matrix")
print(confusion_matrix(y_test,y_pred))
ConfusionMatrixDisplay.from_predictions(y_test,y_pred)
plt.title("Confusion Matrix")
plt.show()
RocCurveDisplay.from_predictions(y_test,y_prob)
plt.title("ROC Curve")
plt.show()