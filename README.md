# Mental Health Emotion Analyzer

A transformer-based NLP web application that analyzes the emotional tone of user-input text. This project uses a fine-tuned Hugging Face transformer model and a Streamlit interface to predict the most likely emotion from written text.

## Overview

This project is designed to classify emotional tone from user-entered text such as journal reflections, messages, or short paragraphs. The application uses a trained transformer model to generate emotion predictions and presents the results through a clean Streamlit web interface.

The app displays:
- the primary predicted emotion
- the top 3 emotion predictions
- confidence scores across all supported emotion classes
- a polished dark-themed UI with sidebar examples

## Features

- Emotion classification from free-text input
- Clean dark-themed Streamlit web UI
- Top predicted emotion with confidence score
- Top 3 emotion predictions
- Confidence distribution chart
- Example prompts in the sidebar
- Cached model loading for faster app performance
- Transformer model weights tracked with Git LFS

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
mental-health-emotion-analyzer/
│
├── app/
│   └── main.py
├── data/
├── models/
│   ├── emotion_model.pkl
│   ├── emotion_transformer/
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   └── vocab.txt
│   ├── train_model.py
│   └── train_transformer.py
├── requirements.txt
├── .gitignore
├── .gitattributes
└── README.md
```

## How It Works

1. The user enters text into the Streamlit app.
2. The tokenizer converts the text into transformer-ready input.
3. The fine-tuned model predicts logits for all emotion classes.
4. Softmax is applied to convert logits into confidence scores.
5. The interface displays:
   - the predicted primary emotion
   - the top 3 emotions
   - a chart showing score distribution across all classes

## Clone the Repository

Since the model weights are tracked with Git LFS, install Git LFS first before cloning.

```bash
git lfs install
git clone https://github.com/vamshikrishnadara/mental-health-emotion-analyzer.git
cd mental-health-emotion-analyzer
```

## Installation

### 1. Create and activate a virtual environment

#### macOS / Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

#### Windows
```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

## Run the App

From the project root folder:

```bash
streamlit run app/main.py
```

Then open the local URL shown in the terminal, usually:

```bash
http://localhost:8501
```

## Model Files

The application loads the trained transformer model from:

```bash
models/emotion_transformer/
```

This folder includes:
- model configuration
- tokenizer files
- vocabulary
- `model.safetensors` model weights

Because the weights file is large, it is stored using Git LFS.

## Example Input

```text
I have been feeling overwhelmed lately, but I am still hopeful that things will get better.
```

## Example Output

- Primary Emotion: optimism
- Top 3 Emotions:
  - optimism
  - nervousness
  - sadness

## Use Cases

- Emotion-aware text analysis
- Mental health related NLP demos
- Portfolio projects for NLP / ML roles
- Human-centered AI applications
- Text-based emotional tone detection

## Important Note

This project is intended for educational and demonstration purposes only.

It is **not** a medical diagnosis tool and should not be used as a substitute for professional mental health advice, therapy, or clinical evaluation.

## Future Improvements

- Add a Clear Text button
- Add confidence percentages in the UI
- Export predictions as CSV
- Add session history
- Deploy on Streamlit Community Cloud
- Add batch text prediction support
- Improve emotion explanations and visualization

## Author

**Vamshi Krishna**
