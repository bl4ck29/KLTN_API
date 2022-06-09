import __init__
from BuildTensorflowModel import BuildTensorflowModel
voc_size = 10000
embedding_vector_feature = 40
sent_len = 5000
name = 'LSTM'
model = BuildTensorflowModel(voc_size, embedding_vector_feature, sent_len, name, compile=True)

import pandas
from EncodeData import EncodeData
from SplitDataset import SplitDataset
import __init__

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
del(data, fakes, reals)

X_train, X_val, X_test, y_train, y_val, y_test = SplitDataset(encoded_text, labels)
import numpy
X_train, X_val, X_test, y_train, y_val, y_test = numpy.array(X_train), numpy.array(X_val), numpy.array(X_test), numpy.array(y_train), numpy.array(y_val), numpy.array(y_test)
import pickle
with open('../assert/TrainData.pkl', 'wb') as file:
    pickle.dump((X_train, y_train, X_val, y_val), file)
with open('../assert/TestData.pkl', 'wb') as file:
    pickle.dump((X_test, y_test), file)

history = model.Fit(X_train, y_train, X_val, y_val)
model.SaveModel('../models/'+name+'/')

import json
with open('../models/'+name+'/history.json', 'w') as file:
    json.dump(history.history, file)