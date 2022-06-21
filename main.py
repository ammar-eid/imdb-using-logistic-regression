import sys
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import dataprep
import evaluate
import time

x_train,x_valid,y_train,y_valid=dataprep.x_train,dataprep.x_valid,dataprep.y_train,dataprep.y_valid

tv = TfidfVectorizer(min_df=0, max_df=1, use_idf=True, ngram_range=(1, 3))
classifier=LogisticRegression()
model = Pipeline([('vectorizer', tv), ('classifier', classifier)])

def train (max_iter=100):
    print(f"working on max_iter {max_iter}")
    print("start training...")
    start=time.time()
    model.fit(x_train,y_train)
    print(f"trained successfully in {time.time()-start}")
def save_model(model_path):
    pickle.dump(model, open(model_path, 'wb'))
    print(f"model saved in {model_path}")

def report(y_true,y_pred):
    evaluate.accuracy(y_true,y_pred)
    evaluate.cmatrix(y_true,y_pred)
    evaluate.creport(y_true,y_pred)
if __name__ == '__main__':
    try:
        train(int(sys.argv[1]))
        save_model("model.sav")
        print("*"*10,"train report","*"*10)
        report(y_train,model.predict(x_train))
        print("*"*10,"validation report","*"*10)
        report(y_valid,model.predict(x_valid))
    except KeyboardInterrupt:
        sys.exit(1)
