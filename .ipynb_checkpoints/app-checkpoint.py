import tensorflow as tf
from tensorflow import keras
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Hiperparámetros
sequence_len = 90

# Cargar modelo y vectorizadores
model = keras.models.load_model("CoreMate_v1")
with open("question_vectorizer.pkl", "rb") as f:
    question_vectorizer = pickle.load(f)
with open("answer_vectorizer.pkl", "rb") as f:
    answer_vectorizer = pickle.load(f)

encoder = model.layers[2]  # Cambia si tenés otro orden
decoder = model.layers[3]

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get")
def get_response():
    user_text = request.args.get("msg")

    vectorized_question = question_vectorizer([user_text])
    encoder_output = encoder(vectorized_question)

    decoded_sentence = ["[START]"]
    for _ in range(sequence_len):
        tokenized_target = answer_vectorizer([" ".join(decoded_sentence)])[:, :-1]
        prediction = decoder([encoder_output, tokenized_target])
        next_token_id = tf.argmax(prediction[0, -1]).numpy()
        next_token = answer_vectorizer.get_vocabulary()[next_token_id]
        if next_token == "[END]":
            break
        decoded_sentence.append(next_token)

    final_response = " ".join(decoded_sentence[1:]).replace("[UNK]", "").strip().capitalize()
    return final_response


if __name__ == "__main__":
    app.run(debug=True, port=8000)

