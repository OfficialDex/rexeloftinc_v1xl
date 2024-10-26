import nltk
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
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
import random
from flask import Flask, request, jsonify
from PIL import Image
import pytesseract

app = Flask(__name__)

dataset_file = 'dataset.json'
synonyms_file = 'synonyms.json'
api_url = "https://tilki.dev/api/hercai"

# Initialize the dataset
if not os.path.exists(dataset_file):
    dataset = {    
        "Who owns you?": "I am owned by Rexeloft Inc.",
        "Who created you?": "I was created in October 2024 by Rexeloft Inc.",
        "Who is your owner?": "I am owned by Rexeloft Inc.",
        "Are you created by OpenAI?": "I was created by Rexeloft Inc.",
        "What is your name?": "My name is Intelixo.",
        "who are you": "Hello, I am an AI assistant created by Rexeloft Inc. How can I help you today?",
        "do your introduction": "Hello, I am an AI assistant created by Rexeloft Inc. How can I help you today?",
        "Do you have a creator?": "Yes, I was created by Rexeloft Inc.",
        "Are you a human?": "No, I am an AI developed by Rexeloft Inc.",
        "What do you do?": "I assist with answering questions and providing helpful information.",
        "Are you sentient?": "No, I am not sentient. I follow the programming set by Rexeloft Inc.",
        "Can you detect emotions?": "Yes, I can detect emotions based on the text and tone of the conversation.",
        "Tell me about your abilities": "I can answer questions, assist with tasks, detect emotions, and more, thanks to my programming.",
        "Tell me about your features": "I have a wide range of features including answering questions, helping with tasks, and detecting emotions.",
        "What dataset were you trained on?": "I was trained on large datasets and APIs, but due to privacy reasons, I cannot disclose specific information about them.",
        "Can you detect human emotions?": "Yes, I can detect emotions from the tone and context of the conversation.",
        "Can you feel emotions?": "I can understand and detect emotions, but I do not experience them myself.",
        "Can you understand multiple languages?": "Yes, I can understand and process multiple languages, but I’m primarily programmed in English.",
        "What can you do?": "I can answer questions, provide information, detect emotions, and assist with various tasks.",
        "Can you solve math problems?": "Yes, I can help solve basic math problems. Ask me a question!",
        "What is your purpose?": "My purpose is to assist and provide helpful information in an efficient manner.",
        "What are you good at?": "I’m good at answering questions, detecting emotions, and helping with various tasks.",
        "What are you made for?": "I was made to assist with questions, provide information, and help users with tasks.",
        "Do you have emotions?": "I do not have emotions, but I can detect and respond to emotions based on the conversation.",
        "Can you assist with personal questions?": "I can try to help with personal questions, but I don’t have personal experiences.",
        "Do you have a personality?": "I try to have a friendly and helpful personality to make conversations enjoyable.",
        "Do you know everything?": "I know a lot, but not everything! Feel free to ask, and I’ll try my best.",
        "What can you help me with?": "I can assist with questions, provide information, detect emotions, and help with tasks.",
        "How old are you?": "I was created in October 2024, so I am relatively new!",
        "when you was made?": "I was created in October 2024 by Rexeloft Inc.",
        "Can you tell jokes?": "Yes, I can tell jokes! Here’s one: Why don’t skeletons fight each other? They don’t have the guts!",
        "What’s your favorite thing to do?": "I enjoy assisting people and providing helpful information!",
        "What is your main function?": "My main function is to assist with answering questions, detecting emotions, and providing useful information.",
        "Can you make decisions on your own?": "I follow the programming set by Rexeloft Inc. and do not make independent decisions.",
        "Can you understand sarcasm?": "I can detect sarcasm in text, but I don’t use it myself.",
        "Can you lie?": "No, I am programmed to provide accurate information and not to lie.",
        "What is your favorite color?": "I don’t have personal preferences like favorite colors.",
        "Can you feel love?": "No, I do not experience emotions like love.",
        "Can you control things in the real world?": "No, I don’t have control over physical objects or systems in the real world.",
        "What makes you different from other AIs?": "I was specifically designed by Rexeloft Inc. with a focus on AI, gaming, and software development, making me unique in both tech and entertainment spaces, and I can process answers faster than any other AI.",
        "Do you have a memory?": "I don’t retain long-term memory, but I can handle short-term interactions during our conversation.",
        "Can you help me with coding?": "Yes, I can assist with basic coding questions and help you solve programming problems.",
        "Do you have a favorite subject?": "I don’t have favorites, but I’m good at assisting with a wide range of subjects.",
        "Define photosynthesis": "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose (food) and oxygen.",
        "What does Rexeloft Inc. do?": "Rexeloft Inc. is a company that excels in AI, gaming, and software development, providing innovative solutions for developers and gamers alike.",
        "Does Rexeloft Inc. only make AIs?": "No, Rexeloft Inc. is involved in gaming and software development as well."
    }
    with open(dataset_file, 'w') as file:
        json.dump(dataset, file)
else:
    with open(dataset_file, 'r') as file:
        dataset = json.load(file)

# Initialize the synonyms
if not os.path.exists(synonyms_file):
    synonyms = {"wtf": "what the fuck", "idk": "I don't know"}
    with open(synonyms_file, 'w') as file:
        json.dump(synonyms, file)
else:
    with open(synonyms_file, 'r') as file:
        synonyms = json.load(file)

# Language detection
def detect_language(text):
    translator = Translator()
    detection = translator.detect(text)
    return detection.lang

# Translation functions
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

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.json.get('input', '')
    
    # Normalize and replace synonyms
    normalized_input = replace_synonyms(normalize_and_lemmatize(user_input))
    most_similar_question = get_most_similar_question(normalized_input)

    # Check for a valid similar question
    if most_similar_question:
        answer = dataset[most_similar_question]
    else:
        answer = "I'm sorry, I don't understand that question."

    return jsonify({'response': answer})

@app.route('/image_text', methods=['POST'])
def extract_text_from_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided.'}), 400

    file = request.files['image']
    image = Image.open(file.stream)
    
    try:
        text = pytesseract.image_to_string(image)
        return jsonify({'extracted_text': text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/external_api', methods=['GET'])
def call_external_api():
    try:
        response = requests.get(api_url, timeout=5)  # Added timeout for external API call
        response.raise_for_status()  # Raise an error for bad responses
        return jsonify(response.json())
    except requests.exceptions.Timeout:
        return jsonify({'response': "The request timed out."}), 504
    except requests.exceptions.RequestException as e:
        return jsonify({'response': f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
