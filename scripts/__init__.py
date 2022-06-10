import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

from gensim.parsing.preprocessing import STOPWORDS
stopwords = list(STOPWORDS)
stopwords.remove('not')
with open('../assert/stopword_en.txt', 'w') as file:
    for word in stopwords:
        file.write(word + '\n')

import sys, os
folders = os.listdir('../')
folders = list(filter(lambda x: '.' not in x, folders))
for folder in folders:
    sys.path.append('../'+folder)