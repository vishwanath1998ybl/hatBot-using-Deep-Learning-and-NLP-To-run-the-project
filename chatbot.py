from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import joblib
import pandas as pd
import numpy as np
import re
import random
import nltk
import string
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.layers import GlobalMaxPooling1D
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
app = Flask(__name__)

replies = []

@app.route("/")
def home():    
    return render_template("chatbot.html") 


@app.route("/",methods=["GET","POST"])
def get_bot_response():    
    if request.method == "POST":
        req = request.form.get('msg')
        print(req)   
        model = load_model('/Users/amanulla/Documents/mindtree/NLP/chatbot/intents_chatbot_tf2_lstm.h5')
        tokenizer = joblib.load('/Users/amanulla/Documents/mindtree/NLP/chatbot/token_chatbot.pkl')
        le = joblib.load('/Users/amanulla/Documents/mindtree/NLP/chatbot/le_chatbot.pkl')
        data = pd.read_csv('/Users/amanulla/Documents/mindtree/NLP/chatbot/chat_data.csv')
        responses = joblib.load('/Users/amanulla/Documents/mindtree/NLP/chatbot/chatbot_responses.pkl')
        prediction_input = req
        texts_p = []
        #removing punctuation and converting to lowercase
        prediction_input = [letters.lower() for letters in prediction_input if letters not in string.punctuation]
        prediction_input = ''.join(prediction_input)
        texts_p.append(prediction_input)
        #tokenizing and padding
        prediction_input = tokenizer.texts_to_sequences(texts_p)
        prediction_input = np.array(prediction_input).reshape(-1)
        prediction_input = pad_sequences([prediction_input],6)
        #getting output from mwodel
        output = model.predict(prediction_input)
        check = output
        #print(max(output[0]))
        output = output.argmax()
        if max(check[0]) < 0.5:
            reply = 'Sorry, i dont know that'
            replies.append((req,reply))
            return render_template("chatbot.html",data=replies) 
        #finding the right tag and predicting
        response_tag = le.inverse_transform([output])[0]
        reply = random.choice(responses[response_tag])
        replies.append((req,reply))
        #print(replies)

    return render_template("chatbot.html",data=replies) 
 
app.run()