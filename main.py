from flask import Flask, request, jsonify
import json
import os
import requests
import re
import random
from googletrans import Translator
from fuzzywuzzy import fuzz
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import wordnet
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from difflib import SequenceMatcher
from functools import lru_cache
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import send_from_directory


app = Flask(__name__)

stemmer = PorterStemmer()
analyzer = SentimentIntensityAnalyzer()
translator = Translator()

emotion_responses = {
    'happy': ["I'm glad to hear that!", "That's awesome!", "Yay!", "very nice!"],
    'sad': ["I'm sorry to hear that.", "I hope things get better soon.", "Stay strong!", "never give up!"],
    'angry': ["Take a deep breath.", "I apologise if I did something wrong.", "sorry if i did anything worng"],
    'neutral': ["Got it.", "Understood.", "Okay!", "Alright!", "Bet"]
}

dataset_file = 'dataset.json'
synonyms_file = 'synonyms.json'
api_url = "https://tilki.dev/api/hercai?soru="

if not os.path.exists(dataset_file):
    dataset = {
        "Who owns you?": "I am owned by Rexeloft Inc.",
        "Who created you?": "I was created in October 2024 by Rexeloft Inc."
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


@lru_cache(maxsize=1000)
def lemmatize_word(word):
    lemmatizer = WordNetLemmatizer()
    return lemmatizer.lemmatize(word)

def detect_language(text):
    detection = translator.detect(text)
    return detection.lang

def translate_to_english(text):
    translation = translator.translate(text, dest='en')
    return translation.text

def translate_from_english(text, lang):
    translation = translator.translate(text, dest=lang)
    return translation.text

def replace_synonyms(text):
    words = text.split()
    return ' '.join(synonyms.get(word.lower(), word) for word in words)

def normalize_and_lemmatize(text):
    words = re.findall(r'\w+', text.lower())
    return ' '.join(lemmatize_word(word) for word in words)

def get_wordnet_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace("_", " "))
    return synonyms

def stem_and_expand_words(words):
    expanded_words = set()
    for word in words:
        stemmed_word = stemmer.stem(word)
        expanded_words.add(stemmed_word)
        expanded_words.update(stemmer.stem(syn) for syn in get_wordnet_synonyms(word))
    return expanded_words

def get_word_similarity(word1, word2):
    return SequenceMatcher(None, word1, word2).ratio()

def get_most_similar_question(question):
    questions = list(dataset.keys())
    if not questions:
        return None

    expanded_question = stem_and_expand_words(question.lower().split())
    highest_ratio, most_similar_question = 0, None

    for q in questions:
        expanded_q = stem_and_expand_words(q.lower().split())
        similarity_ratio = len(expanded_question.intersection(expanded_q)) / len(expanded_question.union(expanded_q))
        combined_score = (similarity_ratio + fuzz.token_set_ratio(question, q) / 100 + 
                          sum(get_word_similarity(w1, w2) for w1 in expanded_question for w2 in expanded_q) /
                          (len(expanded_question) * len(expanded_q)) ) / 3

        if combined_score > highest_ratio:
            highest_ratio, most_similar_question = combined_score, q

    return most_similar_question if highest_ratio > 0.5 else None

def detect_emotion(text):
    sentiment = analyzer.polarity_scores(text)
    if sentiment['compound'] >= 0.5:
        return 'happy'
    elif sentiment['compound'] <= -0.5:
        return 'angry'
    return 'neutral' if sentiment['compound'] == 0 else 'sad'

def respond_based_on_emotion(emotion):
    return random.choice(emotion_responses[emotion])

def query_external_api(question):
    try:
        response = requests.get(api_url + question)
        return response.json().get('cevap') if response.status_code == 200 else None
    except Exception as e:
        print(f"Error querying API: {e}")
        return None

def generate_unique_response(answer):
    return answer

def answer_question(question):
    normalized_question = normalize_and_lemmatize(replace_synonyms(question))
    similar_question = get_most_similar_question(normalized_question)
    if similar_question:
        return generate_unique_response(dataset[similar_question])

    api_response = query_external_api(question)
    if api_response:
        dataset[normalized_question] = api_response
        with open(dataset_file, 'w') as file:
            json.dump(dataset, file, indent=4)
        return generate_unique_response(api_response)
    return "I'm sorry, I don't have an answer for that."

def chatbot_response(user_input):
    user_language = detect_language(user_input)
    translated_input = translate_to_english(user_input) if user_language != 'en' else user_input
    response = answer_question(translated_input)
    return translate_from_english(response, user_language) if user_language != 'en' else response

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    try:
        if request.method == 'GET':
            user_input = request.args.get('message')
            print(f"Received user input via GET: {user_input}")  # Debugging statement
        else:  # POST
            user_input = request.json.get('message')
            print(f"Received user input via POST: {user_input}")  # Debugging statement

        if not user_input:
            print("No input provided")  # Debugging statement
            return jsonify({"error": "No input provided"}), 400

        # Detect emotion from the user input
        emotion = detect_emotion(user_input)
        print(f"Detected emotion: {emotion}")  # Debugging statement

        # Generate an emotional response based on the detected emotion
        emotion_response = respond_based_on_emotion(emotion)
        print(f"Emotion response: {emotion_response}")  # Debugging statement

        # Generate chatbot response
        response = chatbot_response(user_input)
        print(f"Chatbot response: {response}")  # Debugging statement

        # Return the responses as JSON
        return jsonify({"emotion_response": emotion_response, "chatbot_response": response})

    except Exception as e:
        print(f"Error in chat endpoint: {e}")  # Log the error
        return jsonify({"error": "An internal error occurred."}), 500

@app.route('/')
def home():
    return jsonify({"message": "Hello, i am intelixo an ai assistant made by rexeloft inc. how can i help you?"}), 200  

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
