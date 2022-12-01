import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
        This function loads raw data and merges them into 1 dataset
        input: directories of raw data
        ouput: df: merged data
    """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # Merging data
    df = pd.merge(messages,categories, on = 'id')
    return df


def clean_data(df):
    """
        This function cleans data by creating label columns, dropping duplicated rows
        input: df dataframe
        ouput: cleaned data
    """
    # Extract all 0|1 values from category and split to columns
    values = df['categories'].replace('[a-z-_]','',regex = True).str.split(';', expand=True).values

    # Get category column name 
    cols = df.categories[0]
    cols = cols.replace('-1','').replace('-0','').split(';')
    
    # create a dataframe of the 36 individual category columns by categories
    categories = pd.DataFrame([map(int,i) for i in values], columns = cols)
    
    # drop the original categories column from `df`
    df = df.drop('categories',1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis=1)
    
    # drop duplicates
    df.drop_duplicates(inplace = True)
    return df


def save_data(df, database_filename):
    """
        This funtion saves cleaned data
        input: df: cleaned dataframe, database_filename: database directory for saving data
        
    """
    
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('cleaned_data.csv', engine, index=False, if_exists='replace')


def main():
    """
        Runs cleaning data process
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
