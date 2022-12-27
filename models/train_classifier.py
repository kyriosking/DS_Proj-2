# import libraries for data processing
import sys
import pandas as pd
import numpy as np

# Import libraries for database connecting and ML
import sqlite3
import sqlalchemy
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle


nltk.download(['punkt', 'wordnet'])


def load_data(database_filepath):
    """
    This function will load data from Database
    
    :params: database_filepath - string
    :ouputs: X - model features, Y - model target
    """
    # load data from database 
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table("disaster_messages_tb", con=engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    return X,Y


def tokenize(text):
    """
    This function will tokenize given text string.
    
    :params: text - string
    :outputs: clean_tokens - tokenized String after processing
    """
    # Divide text string into multiple parts
    tokens = word_tokenize(text)
    
    # Lemmatization: convert a word into its base form
    # WordNetLemmatizer is an instance to call lemmatizer function
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens=[]
    for token in tokens:
        # Break a sentence into small parts
        # Normalize words by lowering it
        buffer = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(buffer)
        
    return clean_tokens


def build_model():
    """
    This function builds classifier and tunes model using GridSearchCV.
    
    :outputs: cv - Classifier 
    """    
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
        
    parameters = {
        'clf__estimator__n_estimators' : [50, 100]
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=3)
    
    return cv


def evaluate_model(model, X_test, Y_test):
    """
    This function evaluates the performance of model.
    
    :params: model - Classifier, X_test - test dataset, Y_test - labels for X_test
    :outputs: classification report
    """
    y_pred = model.predict(X_test)
    for index, column in enumerate(Y_test):
        print(column, classification_report(Y_test[column], y_pred[:, index]))


def save_model(model, model_filepath):
    """ Exports the final model as a pickle file."""
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """ Builds the model, trains the model, evaluates the model, saves the model."""
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

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
