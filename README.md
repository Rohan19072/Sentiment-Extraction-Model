# Sentiment Analysis Web App

## Overview
This project is a sentiment analysis web application that classifies text into Positive, Negative, or Neutral sentiments. It utilizes a machine learning model trained on a dataset of text and sentiments.

## Project Structure
- `app.py` – Flask web application for user interaction.
- `model.py` – Handles data preprocessing, training, and model persistence.
- `preprocess.py` – Contains functions for cleaning and transforming text data.
- `train.csv` – The dataset used for training the model.
- `model.pkl` – Saved trained model for inference.

## Dependencies
To run this project, install the following Python libraries:

```bash
pip install flask pandas nltk scikit-learn mysql-connector-python neattext
```

## Data Preprocessing
The `preprocess.py` file defines the `pre_process()` function, which performs:
- URL, HTML tag, and emoji removal
- Lowercasing and whitespace normalization
- Stopword removal and stemming
- Hashtag and mention removal

## Model Training
The `model.py` file:
1. Reads the `train.csv` dataset
2. Cleans text using `pre_process()`
3. Stores the cleaned data in a MySQL database
4. Trains a `RandomForestClassifier` using `TfidfVectorizer`
5. Saves the trained model as `model.pkl`

## Web Application
The `app.py` file:
- Loads the trained model
- Accepts user input via a web form
- Preprocesses the text and predicts sentiment
- Displays the classification result

## Running the Application
1. Start the Flask app:
   ```bash
   python app.py
   ```
2. Open `http://127.0.0.1:5000/` in a browser

## Example Prediction
Input: *"I love this product!"*  
Output: **Positive**

## Author
Rohan Dodiya



