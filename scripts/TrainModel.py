import __init__
import pickle, json
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix, recall_score
from tensorflow.keras.models import load_model

from BuildTensorflowModel import BuildTensorflowModel
from EncodeData import main as EncodeDataMain
from Utils import DecodePrediction
def main():
    EncodeDataMain()
    voc_size = 10000
    embedding_vector_feature = 40
    sent_len = 5000
    name = 'BiLSTM'
    model = BuildTensorflowModel(voc_size, embedding_vector_feature, sent_len, name, compile=True)

    with open ('../assert/TrainData.pkl', 'rb') as file:
        X_train, y_train, X_val, y_val = pickle.load(file)
    history = model.Fit(X_train, y_train, X_val, y_val)
    model.SaveModel('../models/'+name+'/')
    with open('../models/'+name+'/history.json', 'w') as file:
        json.dump(history.history, file)

def EvaluateModel(model):
    with open('../assert/TestData.pkl', 'rb') as file:
        X_test, y_test = pickle.load(file)
    print(len(X_test), len(y_test))
    result = model.predict(X_test)
    print(len(result))
    result = [DecodePrediction(val) for val in result]
    dct = {
        'accuracy_score': accuracy_score(y_test, result),
        'f1_score': f1_score(y_test, result),
        'recall_score': recall_score(y_test, result),
        'precision_score': precision_score(y_test, result),
    }
    confusion_matrix = confusion_matrix(test_labels, model_pred)
    return dct, confusion_matrix

def LoadModel(path):
    return load_model(path)

if __name__ == '__main__':
    # main()
    model = LoadModel('../models/LSTM/')
    dct, cm = EvaluateModel(model)
    print(dct)