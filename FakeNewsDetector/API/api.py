from string import punctuation
import nltk

lemma = nltk.wordnet.WordNetLemmatizer()
with open("../assert/stopword_en.txt") as file:
    stopwords = file.read().split("\n")
    stopwords.pop()

from flask import Flask, request, jsonify, Response
from Utils import Fix

api_Detector = Flask(__name__)
api_Detector.config['DEBUG'] = True

@api_Detector.route("/upload")
def UploadTextFile():
    return Response("Notification", status=200)

@api_Detector.route("/fix")
def CleanText(data):
    return jsonify({
        "data" : [' '.join(Fix(text)) for text in data]
    })

if __name__ == '__main__':
    api_Detector.run()