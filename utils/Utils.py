import re, contractions, numpy, pandas

def IsDatetime(text):
    patt_mmddyyyy = "[a-z]*..[0-9]*..[0-9]*$"
    patt_ddmmyyyy = "[0-9]*..[a-z]*..[0-9]*$"
    if (re.match(patt_ddmmyyyy, text)) or (re.match(patt_mmddyyyy, text)):
        return True
    return False

def RemoveWithPattern(patt_start, patt_end, text:str, replace_with):
    while True:
        start_ind = text.find(patt_start)
        if start_ind < 0:
            break
        else:
            end_ind = text.find(patt_end, start_ind)
            if end_ind < 0:
                end_ind = len(text)
            matched = text[start_ind : end_ind]
            text = text.replace(matched, replace_with)
    return text

def RemoveNoneAlphaCharacter(text):
    lst = [text[i] for i in range(len(text))]
    result = []
    for char in lst:
        if char.isalpha():
            result.append(char)
    return ''.join(result)

def Fix(text:str, stopwords:list, punctuation:str, lemma):
    text = text.replace("’", "'")
    for punc in punctuation:
        text = text.replace(punc, '')
    
    patt_textWithNum = '[0-9][a-zA-Z]'
    for group in set(re.findall(patt_textWithNum, text)):
        text = text.replace(group, group[:-1]+' '+group[-1])

    patt_contraction = '[a-z|A-Z]* [s|t|d]'
    for group in set(re.findall(patt_contraction, text)):
        group = group.strip()
        text = text.replace(group, group.replace(' ', "'"))

    text = RemoveWithPattern('https', ' ', text, '')
    text = RemoveWithPattern('@', '', text, '')

    text = contractions.fix(text)

    text = [lemma.lemmatize(word.strip()) for word in text.split()]
    result = []
    for t in text:
        if (t.lower() not in stopwords) and (t.isalpha()):
            result.append(t)
    return result

from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences

def EncodeText(text:numpy.array):
    embedding_vector_feature = 40
    sent_len = 5000
    voc_size = 10000
    converted = [one_hot(content, voc_size) for content in text]
    converted = pad_sequences(converted, padding='pre', maxlen=sent_len)
    return converted

def Preprocess(data, stopwords:list, punctuation:str, lemmatizer):
    lst = [' '.join(Fix(text, stopwords, punctuation, lemmatizer)).lower() for text in data]
    return EncodeText(lst)

def DecodePrediction(val:float):
    if val > 0.5:
        return True
    return False

# from tensorflow.keras.models import load_model
# model_bidirect = load_model('../models/KLTNModel_bidirect/')
# model_unidirect = load_model('../models/KLTNModel/')
# model_other = load_model('../models/KLTNModel_other/')
# def Predict(inp:numpy.array):
#     result = {
#         'bidirect' : model_bidirect,
#         'unidirect' : model_unidirect,
#         'other' : model_other
#         }
#     for key in result.keys():
#         model = result[key]
#         pred = model.predict(inp)
#         result[key] = list(map(DecodePrediction, pred))
#     return result

def ConvertPredictionForTable(lstText:list, dctPrediction:dict):
    result = []
    for i in range(len(lstText)):
        result.append([lstText[i][:20]])
        for key in dctPrediction.keys():
            result[i].append(dctPrediction[key][i])
    return result

def LoadFileToList(path:str):
    try:
        with open(path) as file:
            data = file.read().split("\n")
            if data[-1] == '':
                data.pop()
        return data
    except FileNotFoundError as error:
        raise FileNotFoundError
    else:
        return Response("Catch this exception", status=500)