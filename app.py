from flask import Flask, render_template, request, jsonify, session
from transformers import pipeline
import google.generativeai as ai
import random

# Flask app initialization
app = Flask(__name__)
app.secret_key = "your_secret_key"

# Configure Google Gemini API
API_KEY = 'AIzaSyCIgHspeOtytvBf0_ohZZdt43DUqNJBf2Q'
ai.configure(api_key=API_KEY)
model = ai.GenerativeModel("gemini-pro")
chat = model.start_chat()

# Load pre-trained sentiment analysis pipeline
sentiment_analyzer = pipeline('sentiment-analysis')

# Predefined sentiment mapping
sentiment_mapping = {
    'nallathu': 'POSITIVE',
    'nalla': 'POSITIVE',
    'nanmai': 'POSITIVE',
    'not too bad': 'NEUTRAL',
    'shit': 'BAD WORD',
    'mairu': 'BAD WORD',
}

# Route: Home Page
@app.route("/")
def home():
    session['history'] = []  # Initialize history for each session
    return render_template("index.html")

# Route: Sentiment Analysis
@app.route("/analyze", methods=["POST"])
def analyze_sentiment():
    text = request.json.get("text", "").lower()  # Get text from the frontend
    sentiment, score, response = "NEUTRAL", 0.5, "Neutral feedback detected."

    # Check for custom sentiment mapping
    for word, mapped_sentiment in sentiment_mapping.items():
        if word in text:
            sentiment = mapped_sentiment
            score = 1.0 if sentiment == 'POSITIVE' else 0.0
            response = generate_dynamic_response(sentiment, score)
            break
    else:
        # Use sentiment analyzer for generic text
        result = sentiment_analyzer(text)
        sentiment = result[0]['label']
        score = result[0]['score']
        response = generate_dynamic_response(sentiment, score)

    # Generate Gemini chat response
    chat_message = f"The sentiment of this feedback is {sentiment}. Provide business ideas accordingly."
    ai_response = chat_response(chat_message)

    # Save to session history
    history_entry = {
        "text": text,
        "sentiment": sentiment,
        "response": response,
        "ai_response": ai_response,
    }
    session['history'].append(history_entry)

    return jsonify(history_entry)

# Route: View History
@app.route("/history", methods=["GET"])
def view_history():
    return jsonify(session.get("history", []))

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
