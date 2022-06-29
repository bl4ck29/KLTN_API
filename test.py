import spacy, nltk, pandas
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

text = "Donald Trump just couldn t wish all Americans a Happy New Year and leave it at that. Instead, he had to give a shout out to his enemies, haters and  the very dishonest fake news media.  Donald Trump had just one job to do and he couldn t do it. As our Country rapidly grows stronger and smarter, I want to wish all of my friends, supporters, enemies, haters, and even the very dishonest Fake News Media, a Happy and Healthy New Year,  President Angry Pants tweeted."
NER = spacy.load('en_core_web_sm')
def NETokenize(text):
    NEs = NER(text).ents
    result = []
    # Named entities tokenize
    for chunk in NEs:
        word = chunk.text
        label = chunk.label_
        text = text.replace(word, '')
        result.append((word, label))
    
    text = nltk.word_tokenize(text.lower())
    tagged = nltk.pos_tag(text)
    for word, tag in tagged:
        result.append((word, tag))
    return result

data = pandas.read_csv('path')
data['tokenzied'] = [NETokenize(text) for text in data['content'].tolist()]