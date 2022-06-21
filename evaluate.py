import sys
import pickle
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from  dataprep import x_test as features
from dataprep import y_test as sentiment

model=pickle.load(open("model.sav","rb"))

def creport(y_true,y_pred):
    print(classification_report(y_true, y_pred, target_names=["Positive", "Negative"]))
    print("-*"*20)
def cmatrix(y_true,y_pred):
    print(confusion_matrix(y_true, y_pred))
    print("-*"*20)
def accuracy(y_true,y_pred):
    print(accuracy_score(y_true, y_pred),"%")
    print("-*"*20)

if __name__ == '__main__':
    if (sys.argv[1]=="creport"):
        print("classification report\nprocessing...")
        creport(sentiment,model.predict(features))
    elif (sys.argv[1]=="cmatrix"):
        print("confusion matrix\nprocessing...")
        cmatrix(sentiment,model.predict(features))
    elif (sys.argv[1]=="acc"):
        print("accuracy score\nprocessing...")
        accuracy(sentiment, model.predict(features))

