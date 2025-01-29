# Import libraries
import re
import numpy as np
import pandas as pd
import sqlite3
from sqlalchemy import create_engine
import pickle
import sys

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report


def load_data(database_filepath):
    """
    Load data from SQLite database.

    Args:
        database_filepath (str): Path to SQLite database.

    Returns:
        X (DataFrame): Features (messages).
        Y (DataFrame): Target labels.
        category_names (list): Column names of Y (used in evaluation).
    """
    engine = create_engine(f"sqlite:///{database_filepath}")
    df = pd.read_sql_table('DisasterResponse', engine)
    
    X = df['message']  # Feature column
    Y = df.iloc[:, 4:]  # Target labels
    category_names = Y.columns

    return X, Y, category_names


def tokenize(text):
    """
    Cleans and tokenizes text input.

    Steps:
    1. Detects and replaces URLs with "urlplaceholder".
    2. Tokenizes the text into words.
    3. Lemmatizes tokens, converts to lowercase, and strips whitespace.

    Args:
        text (str): Input text.

    Returns:
        list: List of clean, lemmatized tokens.
    """
    # Detecting URLs
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # Tokenize and lemmatize
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]

    return clean_tokens


def build_model():
    """
    Creates a machine learning pipeline.

    Returns:
        model (Pipeline): A pipeline with CountVectorizer, TfidfTransformer, 
                          and a MultiOutput RandomForestClassifier.
    """
    model = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, ngram_range=(1, 2))),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(min_samples_split=3, n_estimators=150)))
    ])
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluates a trained model and prints classification reports.

    Args:
        model (Pipeline): Trained model.
        X_test (array-like): Test feature data (messages).
        Y_test (DataFrame): True labels.
        category_names (list): List of category names.

    Returns:
        None: Prints classification reports.
    """
    # Make predictions
    Y_pred = model.predict(X_test)
    
    print("\n=== Classification Report for the Model ===\n")
    report = classification_report(Y_test, Y_pred, target_names=Y_test.columns)
    print(report)


def save_model(model, model_filepath):
    """
    Saves the trained model as a pickle file.

    Args:
        model (Pipeline): Trained model.
        model_filepath (str): Path to save the model.
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
