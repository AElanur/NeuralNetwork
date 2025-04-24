from pathlib import Path
from emotion_detection.scripts.mood_predictor import MoodPredictor
import yaml
import random

CONFIG_PATH = Path(__file__).parent.parent

class DataLoader:
    def __init__(self):
        self.mood_predictor = MoodPredictor()
        greeting_config = CONFIG_PATH / "config/greetings.yml"
        expression_config = CONFIG_PATH / "config/expressions.yml"

        with open(greeting_config, encoding='utf-8') as f:
            self.greetings = yaml.safe_load(f)

        with open(expression_config, encoding='utf-8') as f:
            self.expressions = yaml.safe_load(f)

            self.phrase_to_emotion = {
                m["input"].lower(): m["emotion"]
                for m in self.expressions["expressions"]["mappings"]
            }

            self.emotion_responses = self.expressions["expressions"]["bot_responses"]

    def handle_greeting(self, user_message):
        if user_message.lower() in [msg.lower() for msg in self.greetings["greetings"]["user_inputs"]]:
            return random.choice(self.greetings["greetings"]["bot_responses"])
        return None

    def handle_expressions(self, user_message):
        user_msg_lower = user_message.lower()

        if user_msg_lower in self.phrase_to_emotion:
            emotion = self.phrase_to_emotion[user_msg_lower]
            return self.emotion_responses.get(emotion, "Tell me more.")

        predicted_emotion = self.mood_predictor.predict_mood(user_message)
        return self.emotion_responses.get(predicted_emotion, "How does that make you feel?")
