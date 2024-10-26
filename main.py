from flask import Flask, request, jsonify
import nltk
import json
import re
import os
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from googletrans import Translator
from fuzzywuzzy import fuzz
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from functools import lru_cache
from difflib import SequenceMatcher
import random

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

app = Flask(__name__)

dataset_file = 'dataset.json'
synonyms_file = 'synonyms.json'
api_url = "https://tilki.dev/api/hercai?soru="
stemmer = PorterStemmer()
users = {}

emotion_responses = {
    'happy': ["I'm glad to hear that!", "That's awesome!", "Yay!", "very nice!"],
    'sad': ["I'm sorry to hear that.", "I hope things get better soon.", "Stay strong!", "never give up!"],
    'angry': ["Take a deep breath.", "I apologise if i did something worng.", "sorry if i did anything worng"],
    'neutral': ["Got it.", "Understood.", "Okay!", "Alright!", "Bet"]
}

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

    question_words = question.lower().split()
    expanded_question = stem_and_expand_words(question_words)

    highest_ratio = 0
    most_similar_question = None

    for q in questions:
        q_words = q.lower().split()
        expanded_q = stem_and_expand_words(q_words)

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

def respond_based_on_emotion(emotion):
    return random.choice(emotion_responses[emotion])

def query_external_api(question):
    try:
        response = requests.get(api_url + question)
        if response.status_code == 200:
            result = response.json()
            return result.get('cevap')
        else:
            return None
    except Exception as e:
        return None

def generate_unique_response(answer):
    return answer

def answer_question(question):
    normalized_question = normalize_and_lemmatize(replace_synonyms(question))
    similar_question = get_most_similar_question(normalized_question)

    if similar_question:
        answer = dataset[similar_question]
        return generate_unique_response(answer)
    else:
        api_response = query_external_api(question)

        if api_response:
            dataset[normalized_question] = api_response
            with open(dataset_file, 'w') as file:
                json.dump(dataset, file, indent=4)

            return generate_unique_response(api_response)
        else:
            return "I'm sorry, I don't have an answer for that."

@app.route('/')
def home():
    return "Welcome to the AI Assistant! Use the /chat endpoint to interact."

@app.route('/chat', methods=['POST'])
def chat():
    user_id = request.json.get('user_id')
    pass_key = request.json.get('pass_key')
    user_input = request.json.get('message')
    
    if user_id not in users:
        users[user_id] = pass_key

    user_language = detect_language(user_input)
    if user_language != 'en':
        translated_input = translate_to_english(user_input)
    else:
        translated_input = user_input

    response = answer_question(translated_input)

    if user_language != 'en':
        response = translate_from_english(response, user_language)

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
