import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("../data/raw/emotion_data.csv")

def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char.isalnum() or char == ' '])
    return text

data['text'] = data['text'].apply(clean_text)

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

train_data.to_csv("../data/train_data.csv", index=False)
test_data.to_csv("../data/test_data.csv", index=False)

print("Data preprocessed and split!")