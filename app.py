import pickle
import tensorflow as tf
print(tf.__version__)
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from flask import Flask, jsonify, request

app = Flask(__name__)



@app.route('/predict', methods=['POST'])
def post_text_sentiment():

    model = tf.keras.models.load_model("sentiment_analysis_model.h5")

    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    data = request.json
    text = data.get("text")


    text_sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(text_sequence, maxlen=250)

    prediction = model.predict(padded_sequence)
    predicted_class = np.argmax(prediction)

    sentiment = "positive" if predicted_class==1 else "negative"



    return jsonify({"response": sentiment }), 200


if __name__ == "__main__":
    app.run(debug=True)

