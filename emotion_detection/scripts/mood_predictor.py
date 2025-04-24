from pathlib import Path

import joblib

class MoodPredictor:
    def __init__(self):
        base_dir = Path(__file__).parent.parent
        self.model = joblib.load(base_dir / "models/mood_model.pkl")
        self.vectorizer = joblib.load(base_dir / "models/vectorizer.pkl")

    def predict_mood(self, text):
        X = self.vectorizer.transform([text])
        return self.model.predict(X)[0]