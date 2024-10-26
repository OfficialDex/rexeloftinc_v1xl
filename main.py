import nltk
import json
import re
import os
import requests
import time
from googletrans import Translator
from fuzzywuzzy import fuzz
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from functools import lru_cache
from difflib import SequenceMatcher
import logging
from flask import Flask, request, jsonify
from PIL import Image
import pytesseract

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Initialize files and constants
dataset_file = 'dataset.json'
synonyms_file = 'synonyms.json'
api_url = "https://tilki.dev/api/hercai"

# Initialize dataset
if not os.path.exists(dataset_file):
    dataset = {
        "Who owns you?": "I am owned by Rexeloft Inc.",
        "Who created you?": "I was created in October 2024 by Rexeloft Inc.",
        # Add more initial questions and responses as needed...
    }
    with open(dataset_file, 'w') as file:
        json.dump(dataset, file)
else:
    with open(dataset_file, 'r') as file:
        dataset = json.load(file)

# Initialize synonyms
if not os.path.exists(synonyms_file):
    synonyms = {"wtf": "what the heck", "idk": "I don't know"}
    with open(synonyms_file, 'w') as file:
        json.dump(synonyms, file)
else:
    with open(synonyms_file, 'r') as file:
        synonyms = json.load(file)

# Utility functions
def detect_language(text):
    try:
        translator = Translator()
        detection = translator.detect(text)
        return detection.lang
    except Exception as e:
        logging.error(f"Language detection error: {e}")
        return None

def translate_to_english(text):
    try:
        translator = Translator()
        translation = translator.translate(text, dest='en')
        return translation.text
    except Exception as e:
        logging.error(f"Translation to English error: {e}")
        return text

def translate_from_english(text, lang):
    try:
        translator = Translator()
        translation = translator.translate(text, dest=lang)
        return translation.text
    except Exception as e:
        logging.error(f"Translation from English error: {e}")
        return text

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
        logging.debug(f"Found similar question: {most_similar_question} with score {highest_ratio}")
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
            logging.info(f"External API response: {result}")
            return result.get('cevap')
        else:
            logging.warning(f"Failed to get a valid response from the external API. Status code: {response.status_code}")
            return None
    except Exception as e:
        logging.error(f"Error querying external API: {e}")
        return None

def should_store_question(question):
    keywords = ["what", "which", "who", "when", "how", "explain", "define"]
    return any(keyword in question.lower() for keyword in keywords)

def answer_question(question):
    normalized_question = normalize_and_lemmatize(replace_synonyms(question))
    similar_question = get_most_similar_question(normalized_question)

    if similar_question:
        logging.info(f"Answer found in dataset for question: '{question}'")
        return dataset[similar_question]
    else:
        logging.info(f"No answer found in dataset for question: '{question}'")
        return None

def chatbot_response(user_input):
    dataset_answer = answer_question(user_input)

    if dataset_answer:
        return dataset_answer

    logging.info(f"Querying external API for question: '{user_input}'")
    api_response = query_external_api(user_input)
    if api_response and should_store_question(user_input):
        logging.info(f"Storing new question in dataset: '{user_input}'")
        dataset[normalize_and_lemmatize(user_input)] = api_response
        with open(dataset_file, 'w') as file:
            json.dump(dataset, file, indent=4)

    return api_response if api_response else "I'm sorry, I don't have an answer for that."

def extract_text_from_image(image):
    try:
        text = pytesseract.image_to_string(Image.open(image))
        return text.strip()
    except Exception as e:
        logging.error(f"Error extracting text from image: {e}")
        return None

@app.route('/chat', methods=['POST'])
def chat():
    user_id = request.json.get('user_id')
    pass_key = request.json.get('pass_key')
    message = request.json.get('message')
    if request.files:
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
    import os
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
