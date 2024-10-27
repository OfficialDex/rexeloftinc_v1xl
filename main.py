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
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from functools import lru_cache
from difflib import SequenceMatcher
import random

dataset_file = 'dataset.json'
synonyms_file = 'synonyms.json'
api_url = "https://tilki.dev/api/hercai"
conversation_history = []
stemmer = PorterStemmer()

emotion_responses = {
    'happy': ["I'm glad to hear that!", "That's awesome!", "Yay!", "Very nice!"],
    'sad': ["I'm sorry to hear that.", "I hope things get better soon.", "Stay strong!", "Never give up!"],
    'angry': ["Take a deep breath.", "I apologize if I did something wrong.", "Sorry if I did anything wrong"],
    'neutral': ["Got it.", "Understood.", "Okay!", "Alright!", "Bet"]
}

if not os.path.exists(dataset_file):
    dataset = {"Who owns you?": "I am owned by Rexeloft Inc.", "Who created you?": "I was created in October 2024 by Rexeloft Inc."}
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
    expanded_question = set(stemmer.stem(word) for word in question_words)

    highest_ratio = 0
    most_similar_question = None

    for q in questions:
        q_words = q.lower().split()
        expanded_q = set(stemmer.stem(word) for word in q_words)

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
    global conversation_history

    # Check for existing answers in dataset first
    dataset_answer = answer_question(user_input)

    if dataset_answer:
        # Log both the user's question and the dataset answer
        conversation_history.append(f"You: {user_input}")
        conversation_history.append(f"Bot: {dataset_answer}")
        return dataset_answer

    # If no match in dataset, query the external API
    conversation_history.append(f"You: {user_input}")
    history_string = "\n".join(conversation_history)

    api_response = query_external_api(history_string)
    if api_response and should_store_question(user_input):
        dataset[normalize_and_lemmatize(user_input)] = api_response
        with open(dataset_file, 'w') as file:
            json.dump(dataset, file, indent=4)

    conversation_history.append(f"Bot: {api_response if api_response else 'I don’t have an answer.'}")
    return api_response if api_response else "I'm sorry, I don't have an answer for that."

def send_message(message):
    global conversation_history
    response = chatbot_response(message)

    if response:
        print(f"Bot: {response}")
    else:
        print("I'm sorry, I don't have an answer for that.")

    time.sleep(7.5)

def chat():
    global conversation_history

    while True:
        user_input = input("Ask me a question or type 'exit' to leave: ")
        if user_input.lower() == 'exit':
            print("Saving conversation and exiting...")
            with open("conversation_history.json", 'w') as file:
                json.dump(conversation_history, file, indent=4)
            break

        emotion = detect_emotion(user_input)
        emotion_response = respond_based_on_emotion(emotion)
        print(f"{emotion_response}")

        send_message(user_input)

if __name__ == "__main__":
    chat()
