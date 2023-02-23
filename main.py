import jsonlines
import numpy as np
import tensorflow as tf
import json

# load the data from the JSONL file
def load_data(filename):
    with jsonlines.open(filename) as reader:
        conversations = []
        for obj in reader:
            input_text = obj['input']
            output_text = obj['output']
            conversations.append((input_text, output_text))
    return conversations

# train and save the model if it doesn't exist
try:
    model = tf.keras.models.load_model("model.h5")
except:
    data = load_data('data.jsonl')
    x, y = [], []
    for i in data:
        x.append(i[0])
        y.append(i[1])
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(x + y)
    x_seq = tokenizer.texts_to_sequences(x)
    x_pad = tf.keras.preprocessing.sequence.pad_sequences(x_seq, maxlen=max_len, padding="post")
    y_seq = tokenizer.texts_to_sequences(y)
    y_pad = tf.keras.preprocessing.sequence.pad_sequences(y_seq, maxlen=max_len, padding="post")
    y_pad = tf.keras.utils.to_categorical(y_pad, num_classes=vocab_size)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(vocab_size, 64, input_length=max_len),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(vocab_size, activation="softmax")
    ])
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(x_pad, y_pad, epochs=50, batch_size=32, verbose=0)
    model.save("model.h5")

# function to generate the chatbot response
def generate_response(user_input):
    x_seq = tokenizer.texts_to_sequences([user_input])
    x_pad = tf.keras.preprocessing.sequence.pad_sequences(x_seq, maxlen=max_len, padding="post")
    y_seq = np.argmax(model.predict(x_pad), axis=-1)
    y_text = tokenizer.sequences_to_texts(y_seq)[0]
    with open("conversations.jsonl", "a") as f:
        f.write(json.dumps({"input": user_input, "output": y_text}) + "\n")
    return y_text

# interact with the chatbot
print("Welcome to the Chatbot!")
while True:
    user_input = input("You: ")
    response = generate_response(user_input.lower())
    print("Chatbot: " + response)