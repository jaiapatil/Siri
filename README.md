# Siri

A voice-based AI assistant built using Flask, TF-IDF, DailyDialog, Gemini Generative AI, and Web Speech APIs.

## Features

- 🎙️ Voice-based interaction
- 🧠 TF-IDF and Cosine Similarity for local response matching
- 💬 Custom conversational intents using `intents.json`
- 📚 DailyDialog dataset for general conversations
- 🤖 Gemini Generative AI fallback for unknown queries
- 🔊 Text-to-Speech using Web Speech Synthesis API
- 🎤 Speech-to-Text using Web Speech Recognition API
- 🌐 Flask-based backend API
- 💻 Interactive web-based user interface

## Project Structure

```text
Siri/
│
├── app.py
├── index.html
└── intents.json
````

### File Description

| File           | Description                                                        |
| -------------- | ------------------------------------------------------------------ |
| `app.py`       | Flask backend, TF-IDF model, DailyDialog integration and Gemini AI |
| `index.html`   | Web-based user interface and voice interaction                     |
| `intents.json` | Custom intents, patterns and predefined responses                  |

## How It Works

The assistant follows a two-level response mechanism:

```text
User Voice Input
       ↓
Speech Recognition
       ↓
Flask API
       ↓
TF-IDF + Cosine Similarity
       ↓
 ┌─────────────────────┐
 │ Similarity > 0.15   │
 └──────────┬──────────┘
            ↓
 Local Response
(intents.json / DailyDialog)

If similarity ≤ 0.15
            ↓
     Gemini Generative AI
            ↓
      Generated Response
            ↓
     Text-to-Speech
            ↓
       User Response
```

## Technologies Used

* Python
* Flask
* Scikit-learn
* NumPy
* Hugging Face Datasets
* DailyDialog Dataset
* Google Gemini Generative AI
* HTML
* CSS
* JavaScript
* Web Speech API

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/jaiapatil/siri-ai-assistant.git
cd siri-ai-assistant
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install flask scikit-learn numpy datasets google-generativeai
```

### 4. Configure Gemini API

Create a Gemini API key through Google AI Studio.

Set it as an environment variable instead of directly putting the API key in the source code.

Windows:

```bash
set GEMINI_API_KEY=YOUR_API_KEY
```

Linux/macOS:

```bash
export GEMINI_API_KEY=YOUR_API_KEY
```

### 5. Run the Application

```bash
python app.py
```

The application will run at:

```text
http://127.0.0.1:5000
```

Open the URL in a browser and interact with Siri using your microphone.

## Dataset

The project uses the **DailyDialog** conversational dataset through the Hugging Face `datasets` library.

The project also contains a custom `intents.json` file containing predefined conversational intents such as:

* Greetings
* Goodbye
* Name
* Age
* Creator
* Abilities
* Jokes
* Feelings
* Thanks
* Insults

## AI Response System

The assistant first searches its local conversational knowledge using TF-IDF and cosine similarity.

If the similarity score is greater than `0.15`, a response from the local dataset is returned.

If the similarity score is `0.15` or lower, the query is passed to Gemini Generative AI to generate a conversational response.

This allows the system to handle both known and previously unseen queries.

## Future Enhancements

* Improved semantic search using transformer embeddings
* Conversation memory
* Multilingual voice support
* More custom intents
* Improved response evaluation
* User authentication
* Mobile application
* Additional AI models

## License

This project is licensed under the MIT License.

## Author

**Jai Patil**

B.E. Information Technology
Atharva College of Engineering, Mumbai

## Repository

GitHub: https://github.com/jaiapatil/Siri

