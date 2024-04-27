# pylint: disable = W0221
# pylint: disable = W0621

# Import necessary libraries
import os
import pickle
from collections import Counter

import numpy as np
import tensorflow as tf
from keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau
from keras.layers import LSTM, Dense, Dropout, Embedding
from keras.models import Sequential, save_model
from keras.preprocessing.text import Tokenizer
from matplotlib import pyplot
from pandas import Series, read_csv, read_json, read_pickle

from utils import clean_text, download_data, get_pipeline, split_data

RANDOM_SEED = 1  # Set a random seed for reproducibility
BATCH_SIZE = 128  # Set batch size
# Dataset url
URL: str = (
    "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Video_Games.jsonl.gz"
)

download_data(URL)
print("File download complete")

# Read and preprocess data
review_df = read_json(
    os.path.join("data", "Video_Games_5.json"), lines=True, orient="records"
)
review_df = review_df[["overall", "verified", "reviewTime", "reviewText"]]

review_df = review_df.dropna(subset=["reviewText"])
review_df = review_df[review_df["reviewText"].str.strip().str.len() > 0]

verified_df = review_df.loc[review_df["verified"]]
verified_df["targets"] = verified_df["overall"].map({5: 1, 4: 1, 3: 0, 2: 0, 1: 0})
verified_df = verified_df.sample(frac=1.0, random_state=RANDOM_SEED)

inputs, labels = verified_df["reviewText"], verified_df["targets"]

# If preprocessed data doesn't exist, clean and save it
if not os.path.exists(os.path.join("data", "sentiment_features.pkl")):
    inputs = inputs.apply(clean_text)
    print("Text successfully cleaned")

    inputs.to_pickle(os.path.join("data", "sentiment_features.pkl"))
    labels.to_pickle(os.path.join("data", "sentiment_targets.pkl"))

inputs = read_pickle(os.path.join("data", "sentiment_features.pkl"))
labels = read_pickle(os.path.join("data", "sentiment_targets.pkl"))

# Split data into train, validation, and test sets
(train_input, train_label), (valid_input, valid_label), (test_input, test_label) = (
    split_data(inputs, labels)
)

# Tokenize words and prepare data for model training
word_list = [word for doc in train_input for word in doc]
word_count = Counter(word_list)
WORD_FREQUENCY = Series(list(word_count.values()), list(word_count.keys())).sort_values(
    ascending=False
)
n_vocab = (WORD_FREQUENCY >= 25).sum()
sequence_length = train_input.str.len()

tokenizer = Tokenizer(n_vocab, lower=False, oov_token="unk")
tokenizer.fit_on_texts(train_input.to_list())

# Save the tokenizer
with open(os.path.join("tokenizers", "tokenizer.pkl"), "wb") as tokenizer_file:
    pickle.dump(tokenizer, tokenizer_file)

train_input = tokenizer.texts_to_sequences(train_input.to_list())
valid_input = tokenizer.texts_to_sequences(valid_input.to_list())
test_input = tokenizer.texts_to_sequences(test_input.to_list())

# Define model architecture
model = Sequential(
    [
        Embedding(input_dim=n_vocab + 1, output_dim=128, mask_zero=True),
        LSTM(128, return_state=False, return_sequences=False),
        Dense(512, activation=tf.nn.relu),
        Dropout(0.5),
        Dense(1, activation=tf.nn.sigmoid),
    ]
)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# Create TensorFlow datasets for training, validation, and testing
train_ds = get_pipeline(train_input, train_label, batch_size=BATCH_SIZE, shuffle=True)
valid_ds = get_pipeline(valid_input, valid_label, batch_size=BATCH_SIZE, shuffle=True)
test_ds = get_pipeline(test_input, test_label, batch_size=BATCH_SIZE)

# Calculate class weights for imbalanced data
neg_weight = (train_label == 1).sum() / (train_label == 0).sum()

# Define callbacks for model training
os.makedirs("eval", exist_ok=True)
MONITOR_METRIC = "val_loss"
MODE = "min"
print(f"Using {MONITOR_METRIC} metric and mode = {MODE} for EarlyStopping")
print("\n")

csv_logger = CSVLogger(os.path.join("eval", "sentiment_analysis.log"))
lr_callback = ReduceLROnPlateau(patience=3, mode=MODE, min_lr=1e-8)
stopper = EarlyStopping(patience=6, mode=MODE)

# Train the model
model.fit(
    train_ds,
    epochs=10,
    callbacks=[csv_logger, lr_callback, stopper],
    validation_data=valid_ds,
    class_weight={0: neg_weight, 1: 1.0},
)

# Save the trained model
os.makedirs("model", exist_ok=True)
save_model(model, os.path.join("model", "SentimentAnalyser.keras"))

# Evaluate model performance on test data
loss, accuracy = model.evaluate(test_ds)
print(f"Loss: {loss * 100:.2f}%")
print(f"Accuracy: {accuracy * 100:.2f}%")

# Retrieve features, predicted values, and labels for analysis
test_ds = get_pipeline(test_input, test_label, batch_size=BATCH_SIZE)

features, predicted, labels = [], [], []
for feature, label in test_ds:
    features.append(feature)
    predicted.append(model.predict(feature))
    labels.append(label)

features = [text for array in features for text in array.numpy().tolist()]
predicted = tf.concat(predicted, axis=0).numpy()
labels = tf.concat(labels, axis=0).numpy()

# Find indices of most negative and most positive reviews
sorted_pred = np.argsort(predicted.flatten())
min_pred = sorted_pred[:5]
max_pred = sorted_pred[-5:]

# Print most negative and most positive reviews
print("Most negative reviews\n")
print("=" * 50)
for i in min_pred:
    print(" ".join(tokenizer.sequences_to_texts([features[i]])), "\n")

print("\nMost positive reviews\n")
print("=" * 50)
for i in max_pred:
    print(" ".join(tokenizer.sequences_to_texts([features[i]])), "\n")

# Load results
results = read_csv(os.path.join("eval", "sentiment_analysis.log"), index_col=0)
print(results)

# Plot loss
loss_plot = results[["val_loss", "loss"]].plot(
    kind="line", figsize=(8, 4), title="Loss"
)
loss_plot.spines[["top", "right"]].set_visible(False)
loss_plot.set_xlabel("Epochs")
loss_plot.set_ylabel("Loss")
loss_plot.legend(["Validation Loss", "Training Loss"])
loss_plot.figure.savefig(os.path.join("plots", "loss_plot.png"))

# Plot accuracy
accuracy_plot = results[["val_accuracy", "accuracy"]].plot(
    kind="line", figsize=(8, 4), title="Accuracy"
)
accuracy_plot.spines[["top", "right"]].set_visible(False)
accuracy_plot.set_xlabel("Epochs")
accuracy_plot.set_ylabel("Accuracy")
accuracy_plot.legend(["Validation Accuracy", "Training Accuracy"])
accuracy_plot.figure.savefig(os.path.join("plots", "accuracy_plot.png"))

# Show plots
pyplot.show()
