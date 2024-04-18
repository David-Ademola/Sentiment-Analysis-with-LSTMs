# Sentiment Analysis with LSTM

This project implements sentiment analysis on customer reviews using LSTM (Long Short-Term Memory) neural networks. The goal is to classify whether a review is positive or negative based on the text content.

## Introduction

Sentiment analysis is a natural language processing (NLP) technique used to determine the sentiment expressed in a piece of text. In this project, we use LSTM neural networks, a type of recurrent neural network (RNN), to perform sentiment analysis on customer reviews. The LSTM model is trained on a dataset of customer reviews labeled as positive or negative.

## Features

- Text preprocessing: The text data is preprocessed by removing stopwords, lemmatizing words, and tokenizing.
- LSTM model: The main model architecture consists of an embedding layer, an LSTM layer, and fully connected layers with dropout for regularization.
- TensorFlow data pipeline: TensorFlow's `tf.data.Dataset` is used to create an efficient data pipeline for training, validation, and testing.

## Installation

To run this project, you'll need Python 3 and the following libraries:

- TensorFlow
- Keras
- NLTK
- Pandas
- Numpy

You can install these dependencies using pip:

```
pip install requirements.txt
```

## Usage

1. Clone the repository:

```
git clone https://github.com/David-Ademola/Sentiment-Analysis-with-LSTMs.git
```

2. Navigate to the project directory:

```
cd Sentiment-Analysis-with-LSTMs
```

3. Run the main script:

```
python main.py
```

## Dataset

The dataset used for training and evaluation is the [Video Games Reviews dataset](https://jmcauley.ucsd.edu/data/amazon/). It contains reviews from Amazon's video games category and includes information such as review text, rating, and verification status.

## Results

The model's performance is evaluated based on metrics such as loss and accuracy on the test dataset. Additionally, the most negative and most positive reviews are printed for qualitative analysis.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, feel free to open an issue or create a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
