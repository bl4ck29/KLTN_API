import subprocess
subprocess.run('pip3 install -r ../assert/requirements.txt')

import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

from gensim.parsing.preprocessing import STOPWORDS
stopwords = list(STOPWORDS)
stopwords.remove('not')
with open('../assert/stopword_en.txt', 'w') as file:
    for word in stopwords:
        file.write(word + '\n')