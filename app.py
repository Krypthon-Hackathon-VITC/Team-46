from flask import Flask, render_template, jsonify, request

from chatbot import get_response

app = Flask(__name__)

@app.get("/")
def get_index():
    return render_template("base.html")

@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    response = get_response(text)
    message = {"answer":response}
    return jsonify(message)


if __name__ == "__main__":
    app.run(port=8080)
    