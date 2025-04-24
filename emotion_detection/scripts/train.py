from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

SCRIPT_DIR = Path(__file__).parent.absolute()

train_data = pd.read_csv(SCRIPT_DIR / "../data/raw/emotion_data.csv")

vectorizer = TfidfVectorizer(max_features=1000)
X_train = vectorizer.fit_transform(train_data['text'])
y_train = train_data['label']

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

joblib.dump(model, SCRIPT_DIR / "../models/mood_model.pkl")
joblib.dump(vectorizer, SCRIPT_DIR / "../models/vectorizer.pkl")

test_data = pd.read_csv(SCRIPT_DIR / "../data/raw/emotion_data.csv")
X_test = vectorizer.transform(test_data['text'])
y_test = test_data['label']

predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

print("Model trained and saved!")
