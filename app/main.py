import streamlit as st
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# PAGE CONFIG

st.set_page_config(
    page_title="Mental Health Emotion Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CUSTOM CSS

st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #0b1120 0%, #111827 100%);
            color: #e5e7eb;
        }

        section[data-testid="stSidebar"] {
            background: #0f172a;
            border-right: 1px solid rgba(255,255,255,0.06);
        }

        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }

        .hero-card {
            background: linear-gradient(135deg, rgba(30,41,59,0.95), rgba(15,23,42,0.95));
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 22px;
            padding: 30px;
            box-shadow: 0 10px 35px rgba(0,0,0,0.28);
            margin-bottom: 1.25rem;
        }

        .hero-title {
            font-size: 2.2rem;
            font-weight: 800;
            color: #f8fafc;
            margin-bottom: 0.5rem;
        }

        .hero-subtitle {
            font-size: 1.02rem;
            color: #cbd5e1;
            line-height: 1.7;
            margin-bottom: 0.6rem;
        }

        .card {
            background: rgba(15,23,42,0.78);
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 20px;
            padding: 22px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.18);
            margin-bottom: 1rem;
        }

        .section-title {
            font-size: 1.15rem;
            font-weight: 700;
            color: #f8fafc;
            margin-bottom: 0.8rem;
        }

        .result-pill {
            display: inline-block;
            background: linear-gradient(135deg, #2563eb, #7c3aed);
            color: white;
            padding: 10px 18px;
            border-radius: 999px;
            font-size: 1rem;
            font-weight: 700;
            margin-top: 0.5rem;
            margin-bottom: 0.75rem;
        }

        .emotion-desc {
            color: #cbd5e1;
            font-size: 0.98rem;
            line-height: 1.6;
        }

        .metric-box {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 16px;
            padding: 14px;
            text-align: center;
        }

        .metric-label {
            color: #94a3b8;
            font-size: 0.88rem;
            margin-bottom: 4px;
        }

        .metric-value {
            color: #f8fafc;
            font-size: 1.2rem;
            font-weight: 700;
        }

        div[data-testid="stTextArea"] textarea {
            background-color: #0f172a !important;
            color: #f8fafc !important;
            border: 1px solid rgba(255,255,255,0.10) !important;
            border-radius: 16px !important;
            font-size: 16px !important;
            line-height: 1.6 !important;
            min-height: 220px !important;
        }

        div[data-testid="stTextArea"] label {
            color: #e5e7eb !important;
            font-weight: 600 !important;
        }

        .small-note {
            color: #94a3b8;
            font-size: 0.92rem;
            margin-top: 0.4rem;
        }

        hr {
            border: none;
            border-top: 1px solid rgba(255,255,255,0.07);
            margin-top: 1.2rem;
            margin-bottom: 1.2rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# EMOTION LABELS + DESCRIPTIONS

labels = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity",
    "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization", "relief",
    "remorse", "sadness", "surprise", "neutral"
]

emotion_descriptions = {
    "admiration": "A feeling of respect, appreciation, or positive regard toward someone or something.",
    "amusement": "A light, entertained feeling often linked to humor or something enjoyable.",
    "anger": "A strong feeling of frustration, hostility, or irritation.",
    "annoyance": "A milder form of irritation or displeasure.",
    "approval": "A positive judgment or acceptance of something.",
    "caring": "Warmth, concern, kindness, or emotional support toward others.",
    "confusion": "Uncertainty or lack of clarity about what is happening.",
    "curiosity": "Interest in learning, exploring, or understanding something.",
    "desire": "A strong wish or longing for something.",
    "disappointment": "Sadness or frustration because expectations were not met.",
    "disapproval": "A negative judgment or dislike of something.",
    "disgust": "Strong aversion or revulsion toward something unpleasant.",
    "embarrassment": "Self-conscious discomfort, awkwardness, or shame.",
    "excitement": "High-energy enthusiasm, eagerness, or anticipation.",
    "fear": "A sense of threat, worry, or danger.",
    "gratitude": "Thankfulness or appreciation for something positive.",
    "grief": "Deep sorrow, often tied to loss or emotional pain.",
    "joy": "Happiness, delight, or emotional uplift.",
    "love": "Strong affection, closeness, or emotional attachment.",
    "nervousness": "Anxiousness, tension, or unease about something.",
    "optimism": "Hopefulness and expectation of a positive outcome.",
    "pride": "Satisfaction or confidence in oneself or one’s achievements.",
    "realization": "A sudden understanding or moment of clarity.",
    "relief": "Comfort after stress, fear, or uncertainty has reduced.",
    "remorse": "Deep regret or guilt over something done.",
    "sadness": "A low, sorrowful, or unhappy emotional state.",
    "surprise": "A reaction to something unexpected.",
    "neutral": "Emotionally balanced or without a strong detectable feeling."
}

example_prompts = {
    "Stress / Anxiety": "I feel overwhelmed by everything happening right now and I don't know how to handle it anymore.",
    "Hopeful": "Things have been hard lately, but I still believe tomorrow can be better.",
    "Sadness": "I have been feeling empty and emotionally drained for the past few days.",
    "Joy": "I finally finished something I worked so hard on, and I feel amazing.",
    "Anger": "I am frustrated that no one listens even when I clearly explain what I need."
}

# MODEL LOADING

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "emotion_transformer"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_PATH))
    model.eval()
    return tokenizer, model

def analyze_emotion(input_text: str, tokenizer, model):
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)[0].detach().cpu().numpy()

    pred_index = int(np.argmax(probs))
    prediction = labels[pred_index]
    confidence = float(probs[pred_index])

    top3_idx = probs.argsort()[-3:][::-1]
    top3 = [(labels[i], float(probs[i])) for i in top3_idx]

    df = pd.DataFrame({
        "Emotion": labels,
        "Score": probs
    }).sort_values(by="Score", ascending=False)

    return prediction, confidence, top3, df

try:
    tokenizer, model = load_model()
except Exception as e:
    st.error("Model could not be loaded from models/emotion_transformer.")
    st.exception(e)
    st.stop()

# SESSION STATE

if "text_input" not in st.session_state:
    st.session_state.text_input = ""

# SIDEBAR

with st.sidebar:
    st.title(" Analyzer Panel")
    st.markdown("Use this app to detect the dominant emotion in a sentence, paragraph, journal entry, or message.")

    st.markdown("---")
    st.subheader("Quick Examples")

    for label_name, sample_text in example_prompts.items():
        if st.button(label_name, use_container_width=True):
            st.session_state.text_input = sample_text

    st.markdown("---")
    st.subheader("How to Use")
    st.markdown(
        """
        1. Enter or paste text  
        2. Click **Analyze Emotion**  
        3. Review the predicted emotion and top scores
        """
    )

    st.markdown("---")
    st.subheader("Supported Emotions")
    st.caption(f"{len(labels)} emotion classes")

# MAIN LAYOUT

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">Mental Health Emotion Analyzer</div>
        <div class="hero-subtitle">
            A transformer-based web app that analyzes emotional tone from written text.
            Paste a message, journal note, reflection, or paragraph to see the model’s top emotional prediction.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

left_col, right_col = st.columns([1.35, 1], gap="large")

with left_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Enter Text</div>', unsafe_allow_html=True)

    st.session_state.text_input = st.text_area(
        "Text for emotion analysis",
        value=st.session_state.text_input,
        placeholder="Example: I have been feeling mentally exhausted and anxious, but part of me still hopes things will improve soon...",
        height=260,
        label_visibility="collapsed"
    )

    st.markdown(
        '<div class="small-note">Best results usually come from full sentences or short paragraphs rather than single words.</div>',
        unsafe_allow_html=True
    )

    analyze_clicked = st.button("Analyze Emotion", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">About the Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        """
        This model predicts the **most likely emotional tone** in the text.

        It works best when the input:
        - clearly expresses a feeling
        - includes some context
        - is written in natural language
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

# RESULTS

if analyze_clicked:
    user_text = st.session_state.text_input.strip()

    if not user_text:
        st.warning("Please enter some text before clicking Analyze Emotion.")
    else:
        prediction, confidence, top3, df = analyze_emotion(user_text, tokenizer, model)
        description = emotion_descriptions.get(prediction, "No description available.")

        st.markdown("## Results")

        res_left, res_right = st.columns([1.15, 1], gap="large")

        with res_left:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Primary Emotion</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="result-pill">{prediction.capitalize()}</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'<div class="emotion-desc">{description}</div>',
                unsafe_allow_html=True
            )

            st.markdown("<br>", unsafe_allow_html=True)

            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(
                    f"""
                    <div class="metric-box">
                        <div class="metric-label">Confidence</div>
                        <div class="metric-value">{confidence:.3f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with m2:
                st.markdown(
                    f"""
                    <div class="metric-box">
                        <div class="metric-label">Characters</div>
                        <div class="metric-value">{len(user_text)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with m3:
                st.markdown(
                    f"""
                    <div class="metric-box">
                        <div class="metric-label">Words</div>
                        <div class="metric-value">{len(user_text.split())}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Top 3 Emotions</div>', unsafe_allow_html=True)

            for emotion, score in top3:
                st.write(f"**{emotion.capitalize()}**")
                st.progress(float(score))
                st.caption(f"Score: {score:.3f}")

            st.markdown("</div>", unsafe_allow_html=True)

        with res_right:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Confidence Distribution</div>', unsafe_allow_html=True)
            st.bar_chart(df.set_index("Emotion"))
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Model Interpretation</div>', unsafe_allow_html=True)
            st.write(
                f"The model believes the text is primarily **{prediction}** with a confidence score of **{confidence:.3f}**."
            )
            st.write(
                "Use the top-3 emotions to understand secondary emotional signals in the text."
            )
            st.markdown("</div>", unsafe_allow_html=True)
