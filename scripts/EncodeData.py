import sys, os
for folder in os.listdir('../'):
    sys.path.append('../'+ folder)

import pandas, re, contractions, nltk
from string import punctuation
from gensim.parsing.preprocessing import STOPWORDS

from Utils import IsDatetime, Preprocess

def EncodeData(lst_path:list):
    fakes = pandas.read_csv('../dataset/Fake.csv')
    fakes['label'] = 0
    reals = pandas.read_csv('../dataset/True.csv')
    reals['label'] = 1

    data = pandas.concat([fakes, reals], ignore_index=True)
    data['content'] = data['title'] + ' ' + data['text']
    data = data.drop(['title', 'text'], axis=1)
    data = data.dropna()
    data = data.drop_duplicates()
    data = data.reset_index(drop=True)
    columns = list(data.columns)

    data = FilterDatetime(data)

    lemma = nltk.wordnet.WordNetLemmatizer()
    with open('../assert/stopword_en.txt') as file:
        stopwords = file.read().split('\n')

    encoded_text = Preprocess(data['content'].tolist(), stopwords, punctuation, lemma)
    labels = data['label'].tolist()
    return encoded_text, labels

def FilterDatetime(data:pandas.DataFrame):
    conditions = [IsDatetime(row['date'].lower().strip()) for i, row in data.iterrows()]
    return data.loc[conditions]