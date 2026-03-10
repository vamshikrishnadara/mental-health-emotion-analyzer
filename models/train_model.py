import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

data = {
"text":[
"I feel very happy today",
"I am excited about life",
"Today is a wonderful day",
"I feel joyful and optimistic",
"I feel sad and lonely",
"I am feeling depressed",
"I feel hopeless about everything",
"I feel empty and tired",
"This makes me very angry",
"I am furious about what happened",
"I feel frustrated and irritated",
"I feel rage building up",
"I feel stressed about work",
"I am overwhelmed with responsibilities",
"I cannot handle this pressure",
"I feel anxious and stressed",
"I am calm and relaxed",
"I feel positive about the future",
"I am enjoying my day",
"I feel peaceful today"
],

"label":[
"happy",
"happy",
"happy",
"happy",
"sad",
"sad",
"sad",
"sad",
"anger",
"anger",
"anger",
"anger",
"stress",
"stress",
"stress",
"stress",
"happy",
"happy",
"happy",
"happy"
]
}

df = pd.DataFrame(data)

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1,2)
    )),
    ("clf", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

with open("models/emotion_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved!")
