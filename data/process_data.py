# import libraries
import sys
import numpy as np
import pandas as pd
import sqlite3
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Function to load to csv-files and to merge them afterwards

    Args:
        messages_filepath (str): first csv file (messages)
        categories_filepath (str): second csv file (categories)

    Return:
        df (pandas dataframe): Merged messages and categories df, merged on ID
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge on ID
    df = messages.merge(categories, on='id')  
    return df


def clean_data(df):
    """
    Cleans the data:
        - Splits categories into separate columns
        - Converts category values to binary (0 or 1)
        - Replaces value 2 with 0 in the 'related' column
        - Drops duplicates

    Args:
        df (pandas DataFrame): Combined categories and messages DataFrame.

    Returns:
        df (pandas DataFrame): Cleaned DataFrame with split categories.
    """
    # Create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)

    # Select the first row to extract column names
    row = categories.iloc[0]

    # Extract column names by removing the last two characters (-0 or -1)
    category_colnames = [item[:-2] for item in row]

    # Rename the columns of `categories`
    categories.columns = category_colnames

    # Convert category values to numbers (0 or 1)
    for column in categories:
        categories[column] = categories[column].str[-1]  # Keep only the last character
        categories[column] = pd.to_numeric(categories[column])  # Convert to integer

    # Replace incorrect values (2 → 0) in 'related' column
    categories['related'] = categories['related'].replace(2, 0)

    # Drop the original categories column from `df`
    df.drop(columns=['categories'], inplace=True)

    # Concatenate the original dataframe with the new categories dataframe
    df = pd.concat([df, categories], axis=1)

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    """
    Save a DataFrame to an SQLite database.

    Args:
        df (DataFrame): The DataFrame to save.
        database_filename (str): The filename of the SQLite database.
    """
    # Create an SQLite engine
    engine = create_engine(f"sqlite:///{database_filename}")
    
    # Save the DataFrame to a table named 'messages'
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')


def main():
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