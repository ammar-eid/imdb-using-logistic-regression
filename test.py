import sys

import pandas as pd

import dataprep
import pickle
model=pickle.load(open("model.sav","rb"))
if __name__ == '__main__':
        sample=sys.argv[1]
        sample=open(sample,"r").read().strip()
        sample=dataprep.clean(sample)
        sample=pd.Series(sample)
        print(model.predict(sample))





