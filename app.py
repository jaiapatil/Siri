from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import numpy as np
import json
import os

# Import Generative AI
try:
    import google.generativeai as genai
except ImportError:
    print("Please install the Gemini SDK: pip install google-generativeai")
    exit()

# Import the Hugging Face datasets library[cite: 1]
try:
    from datasets import load_dataset
except ImportError:
    print("Please install the datasets library: pip install datasets")
    exit()

app = Flask(__name__)

# ==========================================
# 0. CONFIGURE GEMINI LLM
# ==========================================
# Replace with your actual Google AI Studio API key
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"

if GEMINI_API_KEY != "YOUR_GEMINI_API_KEY":
    genai.configure(api_key=GEMINI_API_KEY)
    # Using flash for incredibly fast voice-conversational responses
    llm_model = genai.GenerativeModel('gemini-1.5-flash')
else:
    llm_model = None
    print("WARNING: Gemini API Key not set. Siri will only use local datasets.")

# ==========================================
# 1. LOAD CORE PERSONA (intents.json)
# ==========================================
print("Loading Core Persona (intents.json)...")
try:
    with open('intents.json', 'r', encoding='utf-8') as file:
        persona_data = json.load(file)
except FileNotFoundError:
    print("Warning: intents.json not found. Siri will only use DailyDialog.")
    persona_data = {"intents": []}

corpus = []        
responses = []     
tags = []          

for intent in persona_data.get("intents", []):
    for pattern in intent["patterns"]:
        corpus.append(pattern.lower())
        responses.append(intent["responses"]) 
        tags.append("persona")

# ==========================================
# 2. LOAD GENERAL KNOWLEDGE (DailyDialog)
# ==========================================
print("Downloading/Loading DailyDialog dataset...")
try:
    dd_dataset = load_dataset("OpenRL/daily_dialog", split="train")
    
    print("Processing DailyDialog conversations...")
    dialogue_count = 0
    
    for dialog in dd_dataset['dialog']:
        dialogue_count += 1
        for i in range(len(dialog) - 1):
            input_text = dialog[i].strip().lower()
            output_text = dialog[i+1].strip()
            
            if input_text and output_text:
                corpus.append(input_text)
                responses.append([output_text]) 
                tags.append("dailydialog")

    print(f"Loaded {dialogue_count} daily conversations.")
except Exception as e:
    print(f"Failed to load DailyDialog: {e}")

print(f"Total conversational patterns learned: {len(corpus)}")

# ==========================================
# 3. TRAIN THE AI MODEL
# ==========================================
print("Training TF-IDF Machine Learning Model. Please wait...")
vectorizer = TfidfVectorizer()
X_training_matrix = vectorizer.fit_transform(corpus)
print("Training Complete. Siri is ready on http://127.0.0.1:5000")

# ==========================================
# 4. THE HYBRID PREDICTION ENGINE
# ==========================================
def generate_ai_response(user_input):
    user_vec = vectorizer.transform([user_input.lower()])
    similarities = cosine_similarity(user_vec, X_training_matrix)
    
    max_sim_index = np.argmax(similarities)
    max_sim_score = similarities[0][max_sim_index]
    
    # LAYER 1: Local Dataset Match (Score > 0.15)[cite: 1]
    if max_sim_score > 0.15:
        matched_responses = responses[max_sim_index]
        matched_tag = tags[max_sim_index]
        
        reply = random.choice(matched_responses)
        reply = reply.replace(" ’ ", "'").replace(" ,", ",").replace(" .", ".")
        
        print(f"[DEBUG] Local Match Score: {max_sim_score:.2f} | Source: {matched_tag}")
        return reply
        
    # LAYER 2: Generative LLM Fallback 
    else:
        print(f"[DEBUG] Low Local Match: {max_sim_score:.2f} -> Routing to Gemini LLM")
        if llm_model:
            try:
                # System prompt keeping her in character based on your intents.json
                prompt = (
                    "You are Siri, a helpful, conversational, and futuristic AI assistant "
                    "developed by Jai Patil. Keep your answer brief, friendly, and under 2 sentences. "
                    "Do not use markdown formatting like asterisks or bold text, as this will be read aloud by a text-to-speech engine. "
                    f"The user says: {user_input}"
                )
                response = llm_model.generate_content(prompt)
                
                # Clean up any residual markdown for the speech synthesizer
                clean_text = response.text.replace("*", "").replace("#", "").strip()
                return clean_text
            except Exception as e:
                print(f"Gemini Error: {e}")
                return "I'm having a little trouble connecting to my neural network right now."
        else:
            return "I'm not quite sure how to respond to that, and my cloud connection is offline."

# ==========================================
# 5. SERVER ROUTES
# ==========================================
@app.route('/')
def index():
    try:
        with open('index.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "index.html not found. Please ensure it is in the same directory.", 404

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    siri_reply = generate_ai_response(user_message)
    return jsonify({"response": siri_reply})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
