import nltk
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

import json
import re
import os
import requests
from googletrans import Translator
from fuzzywuzzy import fuzz
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from functools import lru_cache
from difflib import SequenceMatcher
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import pytesseract

app = Flask(__name__)

dataset_file = 'dataset.json'
synonyms_file = 'synonyms.json'
api_url = "https://tilki.dev/api/hercai"

if not os.path.exists(dataset_file):
    dataset = {
        "Who owns you?": "I am owned by Rexeloft Inc.",
        "Who created you?": "I was created in October 2024 by Rexeloft Inc."
        # ... other entries
    }
    with open(dataset_file, 'w') as file:
        json.dump(dataset, file)
else:
    with open(dataset_file, 'r') as file:
        dataset = json.load(file)

if not os.path.exists(synonyms_file):
    synonyms = {"wtf": "what the fuck", "idk": "I don't know"}
    with open(synonyms_file, 'w') as file:
        json.dump(synonyms, file)
else:
    with open(synonyms_file, 'r') as file:
        synonyms = json.load(file)

def detect_language(text):
    translator = Translator()
    detection = translator.detect(text)
    return detection.lang

def translate_to_english(text):
    translator = Translator()
    translation = translator.translate(text, dest='en')
    return translation.text

def translate_from_english(text, lang):
    translator = Translator()
    translation = translator.translate(text, dest=lang)
    return translation.text

@lru_cache(maxsize=1000)
def lemmatize_word(word):
    lemmatizer = WordNetLemmatizer()
    return lemmatizer.lemmatize(word)

def replace_synonyms(text):
    words = text.split()
    replaced_words = [synonyms.get(word.lower(), word) for word in words]
    return ' '.join(replaced_words)

def normalize_and_lemmatize(text):
    text = text.lower()
    words = re.findall(r'\w+', text)
    lemmatized_words = [lemmatize_word(word) for word in words]
    return ' '.join(lemmatized_words)

def get_word_similarity(word1, word2):
    return SequenceMatcher(None, word1, word2).ratio()

def get_most_similar_question(question):
    questions = list(dataset.keys())
    if not questions:
        return None

    question_words = question.lower().split()
    expanded_question = set(PorterStemmer().stem(word) for word in question_words)

    highest_ratio = 0
    most_similar_question = None

    for q in questions:
        q_words = q.lower().split()
        expanded_q = set(PorterStemmer().stem(word) for word in q_words)

        common_words = expanded_question.intersection(expanded_q)
        similarity_ratio = len(common_words) / len(expanded_question.union(expanded_q))

        fuzzy_ratio = fuzz.token_set_ratio(question, q) / 100
        word_similarity = sum(get_word_similarity(w1, w2) for w1 in expanded_question for w2 in expanded_q) / (len(expanded_question) * len(expanded_q))

        combined_score = (similarity_ratio + fuzzy_ratio + word_similarity) / 3
        if combined_score > highest_ratio:
            highest_ratio = combined_score
            most_similar_question = q

    if highest_ratio > 0.5:
        return most_similar_question
    return None

def detect_emotion(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(text)
    if sentiment_scores['compound'] >= 0.5:
        return 'happy'
    elif sentiment_scores['compound'] <= -0.5:
        return 'angry'
    elif sentiment_scores['compound'] == 0:
        return 'neutral'
    else:
        return 'sad'

def query_external_api(question):
    try:
        params = {'soru': question}
        response = requests.get(api_url, params=params)
        if response.status_code == 200:
            result = response.json()
            return result.get('cevap')
        else:
            return None
    except Exception as e:
        print(f"Error querying API: {e}")
        return None

def should_store_question(question):
    keywords = ["what", "which", "who", "when", "how", "explain", "define"]
    return any(keyword in question.lower() for keyword in keywords)

def answer_question(question):
    normalized_question = normalize_and_lemmatize(replace_synonyms(question))
    similar_question = get_most_similar_question(normalized_question)

    if similar_question:
        return dataset[similar_question]
    else:
        return None

def chatbot_response(user_input):
    dataset_answer = answer_question(user_input)

    if dataset_answer:
        return dataset_answer

    api_response = query_external_api(user_input)
    if api_response and should_store_question(user_input):
        dataset[normalize_and_lemmatize(user_input)] = api_response
        with open(dataset_file, 'w') as file:
            json.dump(dataset, file, indent=4)

    return api_response if api_response else "I'm sorry, I don't have an answer for that."

def extract_text_from_image(image):
    try:
        text = pytesseract.image_to_string(Image.open(image))
        return text.strip()
    except Exception as e:
        return None

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Welcome to the AI Assistant API. Use /chat to interact with me."})

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico')

@app.route('/chat', methods=['POST'])
def chat():
    user_id = request.json.get('user_id')
    pass_key = request.json.get('pass_key')
    message = request.json.get('message')
    
    if request.files:  # Check if an image is uploaded
        image = request.files['image']
        text_from_image = extract_text_from_image(image)
        if text_from_image:
            response = chatbot_response(text_from_image)
            return jsonify({"response": response}), 200
        else:
            return jsonify({"error": "Unable to extract text from image."}), 400

    response = chatbot_response(message)
    return jsonify({"response": response}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
