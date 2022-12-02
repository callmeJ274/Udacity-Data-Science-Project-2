import sys
from sqlalchemy import create_engine
# download necessary NLTK data
import nltk
nltk.download(['punkt', 'wordnet','stopwords'])

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.multioutput import MultiOutputClassifier,MultiOutputRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from nltk.corpus import stopwords
import pickle

# import statements
import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def load_data(database_filepath):
    """
        This function loads cleaned data, defines dependant and independant data and human name labels.
        input: database_filepath: database where the cleaned data saved
        output: X, Y, labels
    """
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table('cleaned_data.csv', engine)
    X = df.loc[:,'message']
    Y = df.iloc[:,4:]
    labels = df.columns[4:]
    return X,Y,labels


def tokenize(text):
    """
        This function tokenizes train data: Remove special characters, Upper case -> lower, Remove stopwords
        input: text
        output: tokens of text
    """
    text = re.sub('[^A-Za-z0-9 ]+', '', text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        if tok not in stopwords.words("english"):
            clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
        This function defines a pipeline for a model: vectorize text, tfidf transform and model
        input: None
        output: pipeline
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('moc', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {
        'clf__estimator__n_estimators' : [50, 100]
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=3)
    
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
        This function evaluates predicted result for test data
        input: model, X_test ( independant features), Y_test (labels), category_names: human label name
        output: prints scores and accuracy score.
    """
    y_pred = model.predict(X_test)
    
    for i in range(len(category_names)):
        print("Category:", category_names[i], "\n", classification_report(Y_test.iloc[:, i].values, y_pred[:, i]))
        print('Accuracy of %25s: %.2f' % (category_names[i], accuracy_score(Y_test.iloc[:, i].values, y_pred[:, i])))


def save_model(model, model_filepath):
    """
        This function saves trained model
        input: trained model, model_filepath: model saving directory
        output: None
    """
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    """
    Runs the whole pipeline    
    """
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
