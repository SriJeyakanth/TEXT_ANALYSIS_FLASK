from flask import Flask, render_template, request, jsonify
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import google.generativeai as ai
import random
import json
import os

# Flask app initialization
app = Flask(__name__)
app.secret_key = "your_secret_key"

# Initialize sentiment analysis models
analyzer_vader = SentimentIntensityAnalyzer()

# Configure Google Gemini API
API_KEY = 'AIzaSyCIgHspeOtytvBf0_ohZZdt43DUqNJBf2Q'
ai.configure(api_key=API_KEY)
model = ai.GenerativeModel("gemini-pro")
chat = model.start_chat()

# JSON file for history
HISTORY_FILE = 'history.json'

sentiment_mapping = {
    'nallathu': 'POSITIVE',
    'nalla': 'POSITIVE',
    'nanmai': 'POSITIVE',
    'pakka': 'POSITIVE',
    'nallavan': 'POSITIVE',
    'sandhosam': 'POSITIVE',
    'aanandham': 'POSITIVE',
    'super': 'POSITIVE',
    'semma': 'POSITIVE',
    'chanceless': 'POSITIVE',
    'magilchi': 'POSITIVE',
    'kettathu': 'NEGATIVE',
    'ketta': 'NEGATIVE',
    'mosam': 'NEGATIVE',
    'drogram': 'NEGATIVE',
    'kobam': 'NEGATIVE',
    'erichal': 'NEGATIVE',
    'aathiram': 'NEGATIVE',
    'koothi': 'BAD WORD',
    'sunni': 'BAD WORD',
    'bastard': 'BAD WORD',
    'fuck': 'BAD WORD',
    'son of bitch': 'BAD WORD',
    'dick': 'BAD WORD',
    'fuck off': 'BAD WORD',
    'punda': 'BAD WORD',
    'pundamavane': 'BAD WORD',
    'othalaka': 'BAD WORD',
    'otha': 'BAD WORD',
    'ngotha': 'BAD WORD',
    'thayoli': 'BAD WORD',
    'kai adi': 'BAD WORD',
    'maireey': 'BAD WORD',
    'kena': 'BAD WORD',
    'polayaadi moone': 'BAD WORD',
    'oombhu': 'BAD WORD',
    'nakku': 'BAD WORD',
    'kundi': 'BAD WORD',
    'soothu': 'BAD WORD',
    'sootha moodu': 'BAD WORD',
    'wakkalaoli': 'BAD WORD',
    'kunju': 'BAD WORD',
    'roudhiram': 'NEGATIVE',
    'paravala': 'NEUTRAL',
    'anyway good': 'NEUTRAL',
    'not bad': 'NEUTRAL',
    'need improvement': 'NEUTRAL',
    'needs improvement': 'NEUTRAL',
    'paravala': 'NEUTRAL',
    'anyway good': 'NEUTRAL',
    'not bad': 'NEUTRAL',
    'need improvement': 'NEUTRAL',
    'good but needs improvement': 'NEUTRAL',
    'needs improvement': 'NEUTRAL',
    'okay': 'NEUTRAL',
    'fine': 'NEUTRAL',
    'alright': 'NEUTRAL',
    'nothing much': 'NEUTRAL',
    'just okay': 'NEUTRAL',
    'could be better': 'NEUTRAL',
    'could improve': 'NEUTRAL',
    'average': 'NEUTRAL',
    'neutral': 'NEUTRAL',
    'alright, I guess': 'NEUTRAL',
    'meh': 'NEUTRAL',
    'kinda okay': 'NEUTRAL',
    'moderate': 'NEUTRAL',
    'not great, not bad': 'NEUTRAL',
    'not perfect': 'NEUTRAL',
    'nothing to complain': 'NEUTRAL',
    'it’s fine': 'NEUTRAL',
    'acceptable': 'NEUTRAL',
    'more or less': 'NEUTRAL',
    'could do better': 'NEUTRAL',
    'just fine': 'NEUTRAL',
    'neither good nor bad': 'NEUTRAL',
    'a bit bland': 'NEUTRAL',
    'kinda dull': 'NEUTRAL',
    'nothing special': 'NEUTRAL',
    'in the middle': 'NEUTRAL',
    'not bad, not good': 'NEUTRAL',
    'more or less alright': 'NEUTRAL',
    'could be worse': 'NEUTRAL',
    'decent': 'NEUTRAL',
    'reasonably okay': 'NEUTRAL',
    'it’s alright': 'NEUTRAL',
    'no big deal': 'NEUTRAL',
    'mediocre': 'NEUTRAL',
    'not too exciting': 'NEUTRAL',
    'not amazing': 'NEUTRAL',
    'not disappointing': 'NEUTRAL',
    'it’s alright, nothing to complain': 'NEUTRAL',
    'just okay-ish': 'NEUTRAL',
    'feels neutral': 'NEUTRAL',
    'not bad, not amazing': 'NEUTRAL',
    'moderately good': 'NEUTRAL',
    'not spectacular': 'NEUTRAL',
    'it’ll do': 'NEUTRAL',
    'it’s decent enough': 'NEUTRAL',
    'so-so': 'NEUTRAL',
    'adequate': 'NEUTRAL',
    'could be improved': 'NEUTRAL',
    'not too bad': 'NEUTRAL',
    'shit': 'BAD WORD',
    'mairu': 'BAD WORD',

}

def read_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as file:
            return json.load(file)
    return []

def write_history(history):
    with open(HISTORY_FILE, 'w') as file:
        json.dump(history, file, indent=4)

# Route: Home Page
@app.route("/")
def home():
    return render_template("work.html")

# Route: Sentiment Analysis
@app.route("/analyze", methods=["POST"])
def analyze_sentiment():
    text = request.json.get("text", "").lower()  # Get text from the frontend
    sentiment = sentiment_mapping.get(text, "NEUTRAL")  # Use predefined mappings
    score = 0.5

    # Use TextBlob for sentiment analysis if not found in predefined mappings
    if sentiment == "NEUTRAL":
        blob = TextBlob(text)
        polarity_tb = blob.sentiment.polarity

        # Use VADER for sentiment analysis
        sentiment_dict = analyzer_vader.polarity_scores(text)
        polarity_vader = sentiment_dict['compound']

        # Ensemble decision: Weighted Averaging
        final_polarity = (polarity_tb + polarity_vader) / 2

        if final_polarity > 0:
            sentiment = "POSITIVE"
        elif final_polarity < 0:
            sentiment = "NEGATIVE"
    
    response = generate_dynamic_response(sentiment, score)

    # Generate Gemini chat response
    chat_message = f"The sentiment of this feedback is {sentiment}. Provide some solutions or suggestions for {sentiment} as business advisor by using more emojis."
    ai_response = chat_response(chat_message)

    # Save to JSON history file
    history = read_history()
    history_entry = {
        "text": text,
        "sentiment": sentiment,
        "response": response,
        "ai_response": ai_response
    }
    history.append(history_entry)
    write_history(history)

    return jsonify(history_entry)

# Route: View History
@app.route("/history", methods=["GET"])
def view_history():
    history = read_history()
    return jsonify(history)

# Route: Delete History
@app.route("/delete", methods=["POST"])
def delete_history():
    entry_id = request.json.get("id")
    history = read_history()
    history = [entry for i, entry in enumerate(history) if i != entry_id]
    write_history(history)
    return jsonify({"status": "success"})

# Helper Functions
def generate_dynamic_response(sentiment, score):
    responses = {
        'POSITIVE': [
            "You're on top of the world! Keep spreading positivity! 🌞",
            "What a wonderful mood! You're glowing with happiness! ✨",
        ],
        'NEGATIVE': [
            "It seems like your text reflects some negative emotions. Stay strong.",
            "It's okay to feel down sometimes. Take a deep breath, and things will improve.",
        ],
        'BAD WORD': [
            "WARNING: Please avoid using inappropriate language. 🚫",
            "ALERT: The words you've used are not acceptable. ⚠️",
        ],
        'NEUTRAL': [
            "Things are steady, nothing to worry about. Just keep going. 😊",
            "No strong emotions today, but you're doing just fine. 🌿",
        ],
    }
    return random.choice(responses.get(sentiment, ["Neutral feedback detected."]))

def chat_response(user_message):
    try:
        response = chat.send_message(user_message)
        return response.text
    except Exception as e:
        return "Error while connecting to Gemini API."

if __name__ == "__main__":
    app.run(debug=True)
