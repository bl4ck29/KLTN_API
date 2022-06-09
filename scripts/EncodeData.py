import pandas, nltk
from string import punctuation
from gensim.parsing.preprocessing import STOPWORDS

from Utils import IsDatetime, Preprocess
def Foo():
    return 'import from EncodeData worked'

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