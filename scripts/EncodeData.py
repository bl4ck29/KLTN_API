import sys, os
folders = os.listdir('../')
folders = list(filter(lambda x: '.' not in x, folders))
for folder in folders:
    sys.path.append('../'+folder)
import pandas, nltk
from string import punctuation

from Utils import IsDatetime, Preprocess
from SplitDataset import SplitDataset

def EncodeData(data:pandas.DataFrame):
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

def main():
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

    encoded_text, labels = EncodeData(data)

    X_train, X_val, X_test, y_train, y_val, y_test = SplitDataset(encoded_text, labels)
    import numpy
    X_train, X_val, X_test, y_train, y_val, y_test = numpy.array(X_train), numpy.array(X_val), numpy.array(X_test), numpy.array(y_train), numpy.array(y_val), numpy.array(y_test)
    import pickle
    with open('../assert/TrainData.pkl', 'wb') as file:
        pickle.dump((X_train, y_train, X_val, y_val), file)
    with open('../assert/TestData.pkl', 'wb') as file:
        pickle.dump((X_test, y_test), file)
main()        