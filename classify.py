import pickle
import re
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def pre_process(data):
    
    # cleaning the data
    pattern = r'[0-9]'
    parse = re.sub(pattern, '' ,data).replace('.','').replace('_','').replace('Agent', '').replace('Customer','')
    clean = re.sub('[\(\[].*?[\)\]]','', parse).replace('{','').replace('}','')

    # tokenization
    tokens = word_tokenize(clean)

    # removing stopwords
    stop_words = set(stopwords.words('english'))
    text_without_stopwords = [t for t in tokens if t not in stop_words]
    # print(text_without_stopwords)

    # lemmatization
    wordnet_lemmatizer = WordNetLemmatizer()
    lemmatize_words = np.vectorize(wordnet_lemmatizer.lemmatize)
    lemmatized_text = ' '.join(lemmatize_words(text_without_stopwords))

    return lemmatized_text


data = open('test_doc.txt').read()

print(data)
clean_data = pre_process(data)
# print(type(clean_data))

with open("trained_model.pkl", 'rb') as file:
    clf = pickle.load(file)

with open("countvect_model.pkl", 'rb') as file:
    count_vect = pickle.load(file)

print(clf)

print(clf.predict(count_vect.transform([clean_data])))
