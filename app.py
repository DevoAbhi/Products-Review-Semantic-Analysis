import pickle
import tensorflow as tf
print(tf.__version__)
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from flask import Flask, jsonify, request

app = Flask(__name__)

# X_test = [[ 0    ,0  ,  0 ,   0 ,   0 ,   0  ,  0 ,   0 ,   0 ,   0  ,  0 ,   0 ,   0 ,   0,
#      0    ,0  ,  0 ,   0 ,   0 ,   0  ,  0 ,   0 ,   0 ,   0  ,  0 ,   0 ,   0 ,   0,
#      0    ,0  ,  0 ,   0 ,   0 ,   0  ,  0 ,   0 ,   0 ,   0  ,  0 ,   0 ,   0 ,   0,
#      0    ,0  ,  0 ,   0 ,   0 ,   0  ,  0 ,   0 ,   0 ,   0  ,  0 ,   0 ,   0 ,   0,
#      0    ,0  ,  0 ,   0 ,   0 ,   0  ,  0 ,   0 ,   0 ,   0  ,  0 ,   0 ,   0 ,   0,
#      0    ,0  ,  0 ,   0 ,   0 ,   0  ,  0 ,   0 ,   0 ,   0  ,  0 ,   0 ,   0 ,   0,
#      0    ,0  ,  0 ,   0 ,   0 ,   0  ,  0 ,   0 ,   0 ,   0  ,  0 ,   0 ,   0 ,   0,
#      0    ,0  ,  0 ,   0 ,   0 ,   0  ,  0 ,   0 ,   0 ,   0  ,  0 ,   0 ,   0 ,   0,
#      0    ,0  ,  0 ,   0 ,   0 ,   0  ,  0 ,   0 ,   0 ,   0  ,  0 ,   0 ,   0 ,   0,
#      0    ,0  ,  0 ,   0 ,   0 ,   0  ,  0 ,   0 ,   0 ,   0  ,  0 ,   0 ,   0 ,   0,
#      0    ,0  ,  0 ,   0 ,   0 ,   0  ,  0 ,   0 ,   0 ,   0  ,  0 ,   0 ,   0 ,   0,
#      0    ,0  ,  0 ,   0 ,   0 ,   0  ,  0 ,   0 ,   0 ,   0  ,  0 ,   0 ,   0 ,   0,
#      0    ,0  ,  0 ,   0 ,   0 ,   0  ,  0 ,   0 ,   0 ,   0  ,  0 ,   0 ,   0 ,   0,
#      0    ,0  ,  0 ,   0 ,   0 ,   0  ,  0 ,   0 ,   0 ,   0  ,  0 ,   0 ,   0 ,   0,
#      0    ,0  ,  0 ,   0 ,   0 ,   0  ,  0 ,   0 ,   0 ,   0  ,  0 ,   0 ,   0 ,   0,
#      0   , 0   , 0   , 0   , 0    ,0   , 0   , 0   , 0 , 152, 2264  ,152 ,2192 , 415,
#   6367 , 200 , 407 ,   5 ,6977 ,  73 ,  38 ,3535 ,2195 , 122 ,  71 ,1227 ,4548 ,5119,
#   7462   ,37 ,1435, 2266 ,3478, 3357 ,2258, 4977,  748  ,  5 ,1349 ,1605]]

# Load the model
def load_model():
    with open('review_sentiment_analysis_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model


@app.route('/predict', methods=['POST'])
def post_text_sentiment():
    # model = load_model()
    model = tf.keras.models.load_model("sentiment_analysis_model.h5")

    # Y_pred = model.predict(X_test)
    # print(np.argmax(Y_pred, axis=1))

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

