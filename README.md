# Mental Health Emotion Analyzer

A transformer-based NLP web application that analyzes the emotional tone of user-input text.  
This project uses a fine-tuned Hugging Face transformer model and a Streamlit interface to predict the most likely emotion from written text.

## Features

- Emotion classification from free-text input
- Clean dark-themed Streamlit web UI
- Top predicted emotion with confidence score
- Top 3 emotion predictions
- Confidence distribution chart
- Example prompts in the sidebar
- Cached model loading for faster app performance

## Tech Stack

- Python
- Streamlit
- PyTorch
- Hugging Face Transformers
- NumPy
- Pandas

## Supported Emotions

The model predicts one of the following 28 emotion classes:

- admiration
- amusement
- anger
- annoyance
- approval
- caring
- confusion
- curiosity
- desire
- disappointment
- disapproval
- disgust
- embarrassment
- excitement
- fear
- gratitude
- grief
- joy
- love
- nervousness
- optimism
- pride
- realization
- relief
- remorse
- sadness
- surprise
- neutral

## Project Structure

```bash
mental-health-nlp-analyzer/
│
├── app/
│   └── main.py
├── data/
├── models/
│   └── emotion_transformer/
├── results/
├── requirements.txt
└── README.md
