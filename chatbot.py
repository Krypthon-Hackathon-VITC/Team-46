import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
keras = tf.keras
K = keras.backend
KL = keras.layers
Lambda, Input, Flatten = KL.Lambda, KL.Input, KL.Flatten
Model = keras.models
KLopt = keras.optimizers.legacy

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words=pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))
model = Model.load_model('chatbot_model.h5')

print(classes)
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        for i,word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    x = np.array([bow])
    res = model.predict(x)[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i ,r in enumerate(res) if r > ERROR_THRESHOLD ]

    results.sort(key = lambda x : x[1] , reverse= True)
    return_list = []
    for r in results :
        return_list.append({'intent' : classes[r[0]],'probablity' : str(r[1])})
    return return_list

def get_response(intents_list,intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

print('Bot is online')
print('write quit to end the conversation')

while True:
    message = input(">>>")
    if message == 'quit' or message == 'quit':
        break
    ints = predict_class(message)
    res = get_response(ints,intents)
    print(res)

