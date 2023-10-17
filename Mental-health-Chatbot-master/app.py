import nltk
nltk.download('popular')
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
import json
import random
import streamlit as st
from keras.models import load_model

# Load the pre-trained model and other necessary data
lemmatizer = WordNetLemmatizer()
model = load_model('model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('texts.pkl', 'rb'))
classes = pickle.load(open('labels.pkl', 'rb'))

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)  
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s: 
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(intents, intent_json):
    tag = intents[0]['intent']
    list_of_intents = intent_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = get_response(ints, intents)
    return res

def main():
    st.title('Chatbot Application')

    # Embed the existing frontend from index.html
    with open("index.html", "r", encoding="utf-8") as f:
        frontend_html = f.read()

    # Render the frontend using the st.components.v1.html component
    st.components.v1.html(frontend_html, height=500, width=800)

    # Add the chatbot functionality
    user_input = st.text_input('You: ')
    if st.button('Send'):
        response = chatbot_response(user_input)
        st.text(f'Bot: {response}')

if __name__ == "__main__":
    main()
