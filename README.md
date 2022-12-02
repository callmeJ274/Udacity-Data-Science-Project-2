# Disaster Response Pipeline Project
## Introduction
The project can be deployed for classifying disaster and analysing the data.
The project create a mocked disaster response pipeline from cleaning data, preprocessing data, training model and evaluating the classification. We're going to use text described the situations, tokenizing and using them for training model. By running web demo, we can see how the model classifies a situation and analyse the data as well

## File Descriptions
app

| - template
| |- master.html # main page of web app
| |- go.html # classification result page of web app
|- run.py # Flask file that runs app

data

|- disaster_categories.csv # data to process
|- disaster_messages.csv # data to process
|- process_data.py # data cleaning pipeline
|- InsertDatabaseName.db # database to save clean data to

models

|- train_classifier.py # machine learning pipeline
|- classifier.pkl # saved model

README.md

## Details
1. ETL Pipeline
data.py creates a data cleaning pipeline that:

Loading raw dataset
Combining 2 dataset
Cleaning the data
Storing cleaned data in a SQLite database.

2. ML Pipeline
train_classifier.py, writes a machine learning pipeline that:

Loadingata for training in SQLite database
Splitting data into training and test sets
Text processing and setting machine learning pipeline
Training and tuning a model by using GridSearchCV
Validating the output
Exporting the final model as a pickle file
## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage
