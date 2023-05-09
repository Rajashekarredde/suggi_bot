# import necessary libraries
from flask import Flask, render_template, request
import json
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
from tensorflow import keras
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
import gensim.downloader as api
import json
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
import os
import urllib.request
from tensorflow import keras
import socket
from gensim.models import KeyedVectors

import gensim.downloader as api
# Load pre-trained word2vec model
model = api.load("word2vec-google-news-300")
lemmatizer = WordNetLemmatizer()

chatbot_model = tf.keras.models.load_model('/Users/rajashekarreddykommula/Downloads/Genie_word2vec.h5')

with open("/Users/rajashekarreddykommula/Downloads/intents.json", "r") as f:
    data = json.load(f)

vocab = set()
classes = []
documents = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        pattern = pattern.rstrip("?")
        tokens = word_tokenize(pattern)
        tokens = [token.lower() for token in tokens if token not in set(stopwords.words("english"))]
        vocab.update(tokens)
        documents.append((tokens, intent["tag"]))
    if intent["tag"] not in classes:
        classes.append(intent["tag"])

X = []
y = []
for doc in documents:
    vecs = []
    for token in doc[0]:
        if token in model.key_to_index:
            vecs.append(model[token])
    if vecs:
        vecs = np.mean(vecs, axis=0)
        X.append(vecs)
        y.append(classes.index(doc[1]))

X = np.array(X)
y = np.array(y)

def vectorize_sentence(sentence, model):
    sentence = sentence.rstrip("?")
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower(), pos='v') for word in sentence_words if word not in set(stopwords.words("english"))]
    vecs = []
    for word in sentence_words:
        if word in model.key_to_index:
            vecs.append(model[word])
    if vecs:
        vecs = np.mean(vecs, axis=0)
    else:
        vecs = np.zeros(model.vector_size)
    return vecs

# Define a function to generate a response
def generate_response(sentence, model):
    vec = vectorize_sentence(sentence, model)
    results = chatbot_model.predict(np.array([vec]))
    results_index = np.argmax(results)
    tag = classes[results_index]
    if results[0][results_index] > 0.5:
        for intent in data["intents"]:
            if intent["tag"] == tag:
                return random.choice(intent["responses"])
    else:
        return "I am evolving constantly. I apologize that I could not help you with your query. Please give us a call at and our team will be happy to assist"

def handle_request(request):
    request_str = request.decode('utf-8')
    print(f"Received request: {request_str}")
    request_lines = request_str.split('\n')
    request_line = request_lines[0]
    method, path, http_version = request_line.split()

    if method == 'OPTIONS':
        print("Handling OPTIONS request")  
        response_str = 'HTTP/1.1 204 No Content\nAccess-Control-Allow-Origin: *\nAccess-Control-Allow-Methods: POST\nAccess-Control-Allow-Headers: Content-Type\n\n'.encode('utf-8')
    elif method == 'POST' and path == '/api/predict':
        print("Handling POST request")  
        content_length = None
        for line in request_lines[1:]:
            if 'Content-Length:' in line:
                content_length = int(line.split()[-1])
        if content_length:
            body = request_lines[-1]
            sentence = json.loads(body)['data']
            response = generate_response(sentence, model)
            response_str = 'HTTP/1.1 200 OK\nContent-Length: {}\nAccess-Control-Allow-Origin: *\n\n{}'.format(len(response), response).encode('utf-8')
        else:
            response_str = 'HTTP/1.1 400 Bad Request\nContent-Length: 0\nAccess-Control-Allow-Origin: *\n\n'.encode('utf-8')
    else:
        response_str = 'HTTP/1.1 404 Not Found\nContent-Length: 0\nAccess-Control-Allow-Origin: *\n\n'.encode('utf-8')

    print(f"Sending response: {response_str.decode('utf-8')}")
    client_connection.sendall(response_str)

server_address = ('localhost', 8000)
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind(server_address)
server_socket.listen(1)
print('Listening on {}:{}'.format(*server_address))

while True:
    client_connection, client_address = server_socket.accept()
    request = client_connection.recv(1024)
    handle_request(request)
    client_connection.close()
