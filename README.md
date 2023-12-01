
# Fake News Detection using Logistic Regression

This repository contains Python code that utilizes Logistic Regression for Fake News Detection. The code implements a machine learning model to classify news articles as real or fake based on their textual content. The process involves several steps such as data preprocessing, feature extraction using TF-IDF Vectorization, model training, and evaluation of accuracy.

## Overview

The provided code performs the following steps:

- **Importing Libraries:** Imports necessary libraries including NumPy, pandas, NLTK, and scikit-learn.
- **Data Loading and Inspection:** Loads a dataset from '/content/train.csv', inspects its shape, displays the initial rows, and counts missing values.
- **Data Preprocessing:** Fills null values, merges 'author' and 'title' columns into a new 'content' column.
- **Text Preprocessing:** Implements stemming on the 'content' column to process textual data for analysis.
- **Feature Extraction:** Converts processed text data into numerical vectors using TF-IDF Vectorizer.
- **Model Training:** Splits the dataset into training and testing sets, initializes a Logistic Regression model, and trains it.
- **Model Evaluation:** Calculates and displays accuracy scores on both training and testing data.
- **Prediction:** Predicts the label of a single sample from the test set and displays the result.

## Usage

1. Clone this repository: `git clone https://github.com/your-username/fake-news-logistic-regression.git`
2. Ensure you have the necessary libraries installed (`numpy`, `pandas`, `nltk`, `scikit-learn`) using `pip`.
3. Run the provided code (`fake_news_detection.py`) in a Python environment to observe the fake news detection process.

Feel free to modify the code, explore different datasets, or enhance the functionality based on your requirements.

## Note

- Ensure the dataset path is correctly specified to match the location of your dataset.
- This code serves as a demonstrative example and may need adaptation for different datasets or requirements.

---

- This project was created following a tutorial as a means of learning and honing skills in machine learning, specifically in the       domain of natural language processing (NLP) and classification tasks.