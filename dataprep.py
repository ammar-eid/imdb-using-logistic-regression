import numpy as np
import pandas as pd
import nltk
from bs4 import BeautifulSoup #for html parsing
import re,string,unicodedata
from nltk.tokenize.toktok import ToktokTokenizer
from sklearn.model_selection import train_test_split


#Tokenization of text
tokenizer=ToktokTokenizer()
#Setting English stopwords
stopword_list=nltk.corpus.stopwords.words('english')
#Removing the html strips
def strip_html(text):
    return BeautifulSoup(text, "html.parser").get_text()

#Removing the square brackets
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

#Removing the noisy text
def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text

#Define function for removing special characters
def rm_specchar(text):
    pattern=r'[^a-zA-z0-9\s]'
    text=re.sub(pattern,'',text)
    return text
#Stemming the text

def simple_stemmer(text):
    ps=nltk.porter.PorterStemmer()
    text= ' '.join([ps.stem(word) for word in text.split()])
    return text

#removing the stopwords
def remove_stopwords(text):
    tokens = tokenizer.tokenize(text)
    filtered_tokens = [token.strip() for token in tokens if token.strip().lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

def clean(data):
    if isinstance(data,str):
        return remove_stopwords(simple_stemmer(rm_specchar(denoise_text(data))))
    else:
        return data.apply(denoise_text).apply(rm_specchar).apply(simple_stemmer).apply(remove_stopwords)

data_path="dataset_updated.csv"
imdb_data=pd.read_csv(data_path).sample(frac=1)

train_x,x_test,train_y,y_test=train_test_split(imdb_data.review,imdb_data.sentiment,test_size=0.2)
x_train,x_valid,y_train,y_valid=train_test_split(train_x,train_y,test_size=0.25)





