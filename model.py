import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import neattext.functions as nfx
import pickle,os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import mysql.connector as connection

os.chdir("d:\Rohan\SEM8\Flask")

df = pd.read_csv("train.csv",encoding='ISO-8859-1')

Data = df.copy()

Data.dropna(subset=['text'],axis=0,inplace=True)

Data.drop(['textID','selected_text'],axis=1,inplace=True)
mapping = {'negative':'Negative','positive':'Positive','neutral':"Neutral"}
Data['sentiment'] = Data['sentiment'].replace(mapping)

conn = connection.connect(host='localhost',database='sql demo',user='root',passwd='',use_pure=True)

cur = conn.cursor()

cols = ",".join([str(i) for i in Data.columns.tolist()])

for i,row in Data.iterrows():
    sql = "INSERT INTO training_data (`text`,`sentiment`) VALUES(%s,%s)"
    cur.execute(sql, tuple(row))
    conn.commit()

conn.close()


def pre_process(text):
    # Remove links
    text = re.sub('http://\S+|https://\S+', '', text)
    text = re.sub('http[s]?://\S+', '', text)
    text = re.sub(r"http\S+", "", text)
 
    # Convert HTML references
    text = re.sub('&amp', 'and', text)
    text = re.sub('&lt', '<', text)
    text = re.sub('&gt', '>', text)

    # Remove new line characters
    text = re.sub('[\r\n]+', ' ', text)
    
    # Remove mentions
    text = re.sub(r'@\w*', '', text)
    
    # Remove hashtags
    text = re.sub(r'#\w*', '', text)

    # Remove multiple space characters
    text = re.sub('\s+',' ', text)
    
    # Convert to lowercase
    text = text.lower()

    # Apply stemming
    ps = PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])

    # Apply NeatText functions
    text = nfx.remove_emojis(text)
    text = nfx.remove_numbers(text)
    text = nfx.remove_emails(text)
    text = nfx.remove_stopwords(text)

    return text

X = Data['text']
Y = Data['sentiment']

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

from sklearn.pipeline import Pipeline

pipe_lr = Pipeline(steps=[('tf',TfidfVectorizer(preprocessor=pre_process)),('lr',RandomForestClassifier())])


pipe_lr.fit(X,Y)


a = " So I really need to put the laptop down & start getting ready for  shindig...But I`ve missed my TwitterLoves all day"
b = pipe_lr.predict([a])
print(b)
print(pipe_lr.score(X_test,Y_test))
pickle.dump(pipe_lr,open("model.pkl","wb"))