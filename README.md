# Disaster Response Pipeline Project
## Introduction
The project can be deployed for classifying disaster and analysing the data.
The project create a mocked disaster response pipeline from cleaning data, preprocessing data, training model and evaluating the classification. We're going to use text described the situations, tokenizing and using them for training model. By running web demo, we can see how the model classifies a situation and analyse the data as well
## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage
