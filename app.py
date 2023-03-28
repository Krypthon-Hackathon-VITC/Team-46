from flask import Flask, render_template, request
import nltk
import numpy as np
import random
import pickle
import json
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
keras = tf.keras
K = keras.backend
KL = keras.layers
Lambda, Input, Flatten = KL.Lambda, KL.Input, KL.Flatten
Model = keras.models
KLopt = keras.optimizers.legacy

#app = Flask(__name__)
app = Flask(__name__)

@app.route("/")
def home():
    return '''
    <!DOCTYPE html>
    <html lang="en">

    <head>
    <meta charset="UTF-8">
    <title>Chatbot</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="ie=edge">
    <link rel="stylesheet" href="{{ url_for('static', filename='styles/style.css') }}">
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.2.1/jquery.min.js"></script>
    </head>

    <body>
    <!-- partial:index.partial.html -->
    <section class="msger">
        <header class="msger-header">
        <div class="msger-header-title">
            <i class="fas fa-bug"></i> Chatbot <i class="fas fa-bug"></i>
        </div>
        </header>

        <main class="msger-chat">
        <div class="msg left-msg">
            <div class="msg-img" style="background-image: url(https://image.flaticon.com/icons/svg/327/327779.svg)"></div>

            <div class="msg-bubble">
            <div class="msg-info">
                <div class="msg-info-name">Chatbot</div>
                <div class="msg-info-time">12:45</div>
            </div>

            <div class="msg-text">
                Hi, welcome to ChatBot! Go ahead and send me a message. 😄
            </div>
            </div>
        </div>

        </main>

        <form class="msger-inputarea">
        <input type="text" class="msger-input" id="textInput" placeholder="Enter your message...">
        <button type="submit" class="msger-send-btn">Send</button>
        </form>
    </section>
    <!-- partial -->
    <script src='https://use.fontawesome.com/releases/v5.0.13/js/all.js'></script>
    <script>

        const msgerForm = get(".msger-inputarea");
        const msgerInput = get(".msger-input");
        const msgerChat = get(".msger-chat");


        // Icons made by Freepik from www.flaticon.com
        const BOT_IMG = "https://image.flaticon.com/icons/svg/327/327779.svg";
        const PERSON_IMG = "https://image.flaticon.com/icons/svg/145/145867.svg";
        const BOT_NAME = "    ChatBot";
        const PERSON_NAME = "You";

        msgerForm.addEventListener("submit", event => {
        event.preventDefault();

        const msgText = msgerInput.value;
        if (!msgText) return;

        appendMessage(PERSON_NAME, PERSON_IMG, "right", msgText);
        msgerInput.value = "";
        botResponse(msgText);
        });

        function appendMessage(name, img, side, text) {
        //   Simple solution for small apps
        const msgHTML = `
    <div class="msg ${side}-msg">
    <div class="msg-img" style="background-image: url(${img})"></div>

    <div class="msg-bubble">
        <div class="msg-info">
        <div class="msg-info-name">${name}</div>
        <div class="msg-info-time">${formatDate(new Date())}</div>
        </div>

        <div class="msg-text">${text}</div>
    </div>
    </div>
    `;

        msgerChat.insertAdjacentHTML("beforeend", msgHTML);
        msgerChat.scrollTop += 500;
        }

        function botResponse(rawText) {

        // Bot Response
        $.get("/get", { msg: rawText }).done(function (data) {
            console.log(rawText);
            console.log(data);
            const msgText = data;
            appendMessage(BOT_NAME, BOT_IMG, "left", msgText);

        });

        }


        // Utils
        function get(selector, root = document) {
        return root.querySelector(selector);
        }

        function formatDate(date) {
        const h = "0" + date.getHours();
        const m = "0" + date.getMinutes();

        return `${h.slice(-2)}:${m.slice(-2)}`;
        }



    </script>

    </body>

    </html>'''

@app.route("/get")
def chatbot_response():
    model = Model.load_model('chatbot_model.h5')
    intents = json.loads(open('intents.json').read())
    words = pickle.load(open('words.pkl', 'rb'))
    classes = pickle.load(open('classes.pkl', 'rb'))

    def clean_sentence(sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [word.lower() for word in sentence_words]
        return sentence_words

    def bag_of_words(sentence):
        sentence_words = clean_sentence(sentence)
        bag = [0] * len(words)
        for w in sentence_words:
            for i, word in enumerate(words):
                if word == w:
                    bag[i] = 1
        return np.array(bag)

    def predict_class(sentence):
        bow = bag_of_words(sentence)
        res = model.predict(np.array([bow]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
        return return_list

    def get_response(intents_list, intents_json):
        tag = intents_list[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                break
        return result

    message = request.args.get('msg')
    ints = predict_class(message)
    res = get_response(ints, intents)
    return res

if __name__ == "__main__":
    app.run(debug=True)
