# Import necessary modules
import os
import pickle

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

from utils import clean_text

MAX_LEN: int = 50
THRESHOLD: float = 0.20

# Load pretrained model
model = load_model(os.path.join("model", "SentimentAnalyser.keras"))

# Load tokenizer
with open(os.path.join("tokenizers", "tokenizer.pkl"), "rb") as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

user_input = input("Enter your product review: ")  # Prompt user input
cleaned_input = clean_text(user_input)
# Tokenize input with the same tokenizer used during training
tokenized_input = tokenizer.texts_to_sequences([cleaned_input])
padded_input = pad_sequences(tokenized_input, MAX_LEN)  # Pad input to a fixed length

# Predict sentiment
prediction = model.predict(padded_input)

# Logic to determine the sentiment label
if prediction > THRESHOLD:
    print("Thank you for your positive review of our product! 😊")
elif 0.10 <= prediction <= THRESHOLD:
    print("We'll work on improving our product. Thank you for your feedback")
else:
    print("We're so sorry to hear that 😥.")
