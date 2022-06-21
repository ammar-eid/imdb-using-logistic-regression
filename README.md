# imdb-using-logistic-regression
## steps
### 1- preparing data assume data is presented download through ( https://www.kaggle.com/code/lakshmi25npathi/sentiment-analysis-of-imdb-movie-reviews/data )
##### 1.1- html parsing
##### 1.2- remove any text between square brackets
##### 1.3- remove all special charachtars
##### 1.4- stemming all what can be steemed (e.g played => play)
##### 1.5- remove all stop-words
### 2- splitting data train --> 60% valid --> 20% test --> 60%
### 3- train data through a Pipeline model (TfidfVectorizer,Logistic Regression)
### 4- save the model in a file named (model.sav)
### 5- display metrics for both training and validstion-set
#### 5.1 Confusion matrix
#### 5.2 Classification report
#### 5.3 accuracy score
## additional features
### testing through test.py
#### 1- single input entered manual
#### 2- single input in a file in a .txt form
#### 3- multiple input in a .csv form
### evaluating throught evaluate.py
#### 1. Confusion matrix
#### 2. Classification report
#### 3. accuracy score 
### working in a CLI form
#### python train.py max-iter
#### python test.py text | text.txt | text.csv
#### python evaluate.py cmatrix | creport | acc   ---> confusion matrix , classification report , accuracy report
