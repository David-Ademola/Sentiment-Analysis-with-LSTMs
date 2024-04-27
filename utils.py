# pylint: disable = W0102
import gzip
import os
import re
import shutil
from string import punctuation

import requests
import tensorflow as tf
from pandas import Series
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import nltk
from nltk import WordNetLemmatizer, pos_tag

RANDOM_SEED = 1

nltk.download("averaged_perceptron_tagger", download_dir="nltk")
nltk.download("wordnet", download_dir="nltk")
nltk.download("omw-1.4", download_dir="nltk")
nltk.download("stopwords", download_dir="nltk")
nltk.download("punkt", download_dir="nltk")
nltk.data.path.append(os.path.abspath("nltk"))


def download_data(url: str) -> None:
    """Downloads the dataset from a given url"""
    # Check if the file already exists
    if not os.path.exists(os.path.join("data", "Video_Games.jsonl.gz")):
        # Get the file from web
        r = requests.get(url, stream=True, timeout=60)
        total_size = int(r.headers.get('content-length', 0))

        # Create the data directory if it doesn't exist
        if not os.path.exists("data"):
            os.mkdir("data")

        # Write to a file with progress bar
        with open(os.path.join("data", "Video_Games.jsonl.gz"), "wb") as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=url.split('/')[-1]) as pbar:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
    else:
        print("The tar file already exists.")

    # Extract the file if needed
    if not os.path.exists(os.path.join("data", "Video_Games.jsonl")):
        with gzip.open(os.path.join("data", "Video_Games.jsonl.gz"), "rb") as f_in:
            with open(os.path.join("data", "Video_Games.jsonl"), "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
    else:
        print("The extracted data already exists")


def clean_text(doc: str) -> list[str]:
    """A function that cleans a given text document"""
    # Define stopwords and lemmatizer
    stop_words = set(stopwords.words("english")) - {"not", "no"}
    lemmatizer = WordNetLemmatizer()

    # Preprocessing steps
    doc = doc.lower()
    doc = doc.replace("n't", " not ")
    doc = re.sub(r"(?:\'ll |\'re |\'d |\'ve)", " ", doc)
    doc = re.sub(r"/d+", "", doc)

    # Tokenize and filter stopwords
    tokens = [
        word
        for word in word_tokenize(doc)
        if word not in stop_words and word not in punctuation
    ]

    # Part of speech tagging and lemmatization
    pos_tags = pos_tag(tokens)
    cleaned_text = [
        (
            lemmatizer.lemmatize(word, part_of_speech[0].lower())
            if part_of_speech[0] in "NV"
            else word
        )
        for word, part_of_speech in pos_tags
    ]

    return cleaned_text


def split_data(features: Series, targets: Series, train_fraction: float = 0.8):
    """Splits a given dataset into training, validation and test sets"""
    # Separate indices of negative and positive data points
    neg_indices = Series(targets.loc[(targets == 0)].index)
    pos_indices = Series(targets.loc[(targets == 1)].index)

    # Determine test set size
    n_test = int(
        min([len(neg_indices), len(pos_indices)]) * ((1 - train_fraction) / 2.0)
    )

    # Split indices into train, validation, and test sets
    neg_test_inds = neg_indices.sample(n=n_test, random_state=RANDOM_SEED)
    neg_valid_inds = neg_indices.loc[~neg_indices.isin(neg_test_inds)].sample(
        n=n_test, random_state=RANDOM_SEED
    )
    neg_train_inds = neg_indices.loc[
        ~neg_indices.isin(neg_test_inds.tolist() + neg_valid_inds.tolist())
    ]

    # Similarly split positive indices
    pos_test_inds = pos_indices.sample(n=n_test, random_state=RANDOM_SEED)
    pos_valid_inds = pos_indices.loc[~pos_indices.isin(pos_test_inds)].sample(
        n=n_test, random_state=RANDOM_SEED
    )
    pos_train_inds = pos_indices.loc[
        ~pos_indices.isin(pos_test_inds.tolist() + pos_valid_inds.tolist())
    ]

    # Extract train, validation, and test datasets
    train_features = features.loc[
        neg_train_inds.tolist() + pos_train_inds.tolist()
    ].sample(frac=1.0, random_state=RANDOM_SEED)
    train_target = targets.loc[
        neg_train_inds.tolist() + pos_train_inds.tolist()
    ].sample(frac=1.0, random_state=RANDOM_SEED)

    valid_features = features.loc[
        neg_valid_inds.tolist() + pos_valid_inds.tolist()
    ].sample(frac=1.0, random_state=RANDOM_SEED)
    valid_targets = targets.loc[
        neg_valid_inds.tolist() + pos_valid_inds.tolist()
    ].sample(frac=1.0, random_state=RANDOM_SEED)

    test_features = features.loc[
        neg_test_inds.tolist() + pos_test_inds.tolist()
    ].sample(frac=1.0, random_state=RANDOM_SEED)
    test_targets = targets.loc[neg_test_inds.tolist() + pos_test_inds.tolist()].sample(
        frac=1.0, random_state=RANDOM_SEED
    )

    print(f"Training data: {len(train_features)}")
    print(f"Validation data: {len(valid_features)}")
    print(f"Test data: {len(test_features)}")

    return (
        (train_features, train_target),
        (valid_features, valid_targets),
        (test_features, test_targets),
    )


def get_pipeline(
    text_seq: list[list[int]],
    outputs: Series,
    batch_size: int = 64,
    bucket_boundaries: list[int] = (5, 15),
    max_length: int = 50,
    shuffle: bool = False,
):
    """Data pipeline that converts sequences to batches of data"""
    data_seq = [[b] + a for a, b in zip(text_seq, outputs)]
    tf_data = tf.ragged.constant(data_seq)[:, :max_length]
    text_ds = tf.data.Dataset.from_tensor_slices(tf_data)

    # Bucketing based on sequence length
    bucket_fn = tf.data.experimental.bucket_by_sequence_length(
        lambda x: tf.cast(tf.shape(x)[0], tf.int32),
        bucket_boundaries=bucket_boundaries,
        bucket_batch_sizes=[batch_size] * 3,
        padded_shapes=None,
        padding_values=0,
    )
    text_ds = text_ds.map(lambda x: x).apply(bucket_fn)

    if shuffle:
        text_ds = text_ds.shuffle(buffer_size=10 * batch_size)

    # Separate features and labels
    text_ds = text_ds.map(lambda x: (x[:, 1:], x[:, 0]))

    return text_ds
