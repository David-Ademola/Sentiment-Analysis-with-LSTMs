# Sentiment Analysis with LSTM

This project is a sentiment analysis tool that utilizes a Long Short-Term Memory (LSTM) neural network to classify the sentiment of product reviews as positive or negative.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Example](#example)
- [Dataset](#dataset)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Sentiment analysis is a natural language processing (NLP) task that involves determining the sentiment expressed in a piece of text, such as a review or comment. This project aims to automatically classify the sentiment of product reviews as either positive or negative using deep learning techniques.

## Features

- Preprocesses text data to remove noise and standardize input.
- Utilizes an LSTM neural network for sentiment classification.
- Provides a console app for easy sentiment analysis of user-inputted text.
- Includes model training and evaluation scripts for customization and further development.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/David-Ademola/Sentiment-Analysis-with-LSTMs
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Train the sentiment analysis model:

   ```bash
   python main.py
   ```

2. Use the console app to analyze the sentiment of text:

   ```bash
   python app.py
   ```

## Example

```
# Analyze sentiment using the console app
Enter your product review: This game is amazing! I love it!
>>> Thank you for your positive review of our product! 😊
```

## Dataset

The dataset used for training and evaluation is the [Video Games Reviews dataset](https://www.kaggle.com/datasets/drshoaib/amazon-videogames-reviews). It contains reviews from Amazon's video games category and includes information such as review text, rating, and verification status.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, feel free to open an issue or create a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
