import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model_lstm.h5")

@st.cache_resource
def load_tokenizer():
    with open("tokenizer1.pickle", "rb") as handle:
        return pickle.load(handle)

model = load_model()
tokenizer = load_tokenizer()


st.title("Twitter Sentiment Classifier")
st.write("Enter a tweet below and find out its predicted sentiment class.")


tweet = st.text_area("Enter a tweet:")


if st.button("Predict"):
    if tweet.strip() != "":
        seq = tokenizer.texts_to_sequences([tweet])
        padded = pad_sequences(seq, maxlen=100)  # ⚠️ set this to the same maxlen used in training
        prediction = model.predict(padded)
        pred_class = prediction.argmax(axis=1)[0]

        # Optional: Map numeric class to labels
        label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        st.success(f"Predicted Sentiment: {label_map.get(pred_class, pred_class)}")
    else:
        st.warning("Please enter a tweet to classify.")








