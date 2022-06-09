import sys, os
for folder in os.listdir('../'):
    sys.path.append('../'+ folder)

from string import punctuation
import nltk


lemma = nltk.wordnet.WordNetLemmatizer()
with open('../assert/stopword_en.txt') as file:
    stopwords = file.read().split('\n')

from flask import Flask, Response, request, render_template, redirect, url_for
from Utils import Preprocess, Predict, LoadFileToList, ConvertPredictionForTable

# API config
api_FakeNewsDetector = Flask(__name__)
api_FakeNewsDetector.config['DEBUG'] = True

@api_FakeNewsDetector.route("/")
def Home():
    return render_template("Homepage.html")

def ShowPrediction(arg):
    return render_template("Prediction.html", items=arg)

@api_FakeNewsDetector.route("/submit", methods=['POST'])
def Submit():
    if request.method == 'POST':
        path_src = request.form.get('path_src')
        if path_src=='':
            path_src = None
        text = request.form.get('text')
        if text == '':
            text = None

        if path_src and text:
            return Response('Too many input type. Choose one: UPLOADED FILE or TEXT', status=400)
        elif path_src or text:
            data = None
            if path_src:
                data = LoadFileToList(path_src)
            elif text:
                data = [text]
            encoded = Preprocess(data, stopwords, punctuation, lemma)
            result = Predict(encoded)
            lst = ConvertPredictionForTable(data, result)
            return ShowPrediction(lst)
        else:
            return Response('No field was filled', status=400)

api_FakeNewsDetector.run(host='0.0.0.0')