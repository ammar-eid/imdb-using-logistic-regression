import sys
import pandas as pd
import dataprep
import pickle
if __name__ == '__main__':
        try:
                i_data=str(sys.argv[1])
                if (i_data=="text"):
                        print("write your review")
                        sample = dataprep.clean(input())
                        sample = pd.Series(sample)
                elif(".txt" in i_data):
                        sample=open(i_data,"r").read().strip()
                        sample = dataprep.clean(sample)
                        sample = pd.Series(sample)
                elif (".csv" in i_data):
                        df=pd.read_csv(i_data)
                        df=dataprep.clean(df[df.columns[0]])
                else:
                        print("sorry not recognized")
                        print("you have 3 options :\n=>load text_file e.g sample.txt\n=>load csv_file e,g sample.csv")
                        print("=> write your input through text")
                        sys.exit(1)
                model = pickle.load(open("model1.sav", "rb"))
                print(model.predict(sample))
        except FileNotFoundError as ex:
                print("cannot find a file or the model have been not trained yet!!")
                print(ex)
        except Exception as ex:
                print(ex)





