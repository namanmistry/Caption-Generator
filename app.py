import numpy as np
import pandas as pd
import cv2
import os
from glob import glob
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Flatten,Input, Convolution2D, Dropout, LSTM, TimeDistributed, Embedding, Bidirectional, Activation, RepeatVector,Concatenate
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications import ResNet50
import pickle
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from flask import Flask, render_template,request
import flask
from werkzeug.utils import secure_filename

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#defining the previous model
incept_model = ResNet50(include_top=True)
last = incept_model.layers[-2].output
modele = Model(inputs=incept_model.input, outputs=last)

model = load_model("model.h5")

pickle_open = open("count_words.pkl","rb")
count_words = pickle.load(pickle_open)
pickle_open.close()

pickle_open = open("inv_dict.pkl","rb")
inv_dict = pickle.load(pickle_open)
pickle_open.close()

#predicting
def getImage(image):
    
    test_img_path = image
    print(image)
    test_img = cv2.imread(test_img_path)
    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

    test_img = cv2.resize(test_img, (224,224))

    test_img = np.reshape(test_img, (1,224,224,3))
    return test_img

def predict(image):
    
    test_feature = modele.predict(getImage(image)).reshape(1,2048)
    print(image)
    test_img_path =  image
    test_img = cv2.imread(test_img_path)
    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    text_inp = ['startofseq']
    count = 0
    caption = ''
    while count < 25:
        count += 1

        encoded = []
        for i in text_inp:
            encoded.append(count_words[i])

        encoded = [encoded]

        encoded = pad_sequences(encoded, padding='post', truncating='post', maxlen=34)


        prediction = np.argmax(model.predict([test_feature, encoded]))

        sampled_word = inv_dict[prediction]

        caption = caption + ' ' + sampled_word
            
        if sampled_word == 'endofseq':
            break

        text_inp.append(sampled_word)
    return caption

app = Flask(__name__)
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/after",methods = ['POST', 'GET'])
def after():
    imagefile = request.files['myFile']
    imagefile.save(secure_filename(imagefile.filename))
    print(imagefile.filename)
    predicted_words = predict(imagefile.filename)
    os.remove(imagefile.filename)
    return f"{predicted_words}"
if __name__ == "__main__":
    app.run(debug=True,port=8000, host='0.0.0.0')