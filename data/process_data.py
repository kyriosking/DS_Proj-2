# Import Libraries
import sys
import pandas as pd
import numpy as np
import sqlite3
import sqlalchemy
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    This function loads and merges 2 datasets.
    
    :params: messages_filepath - disaster_messages.csv dir path
             categories_filepath: disaster_categories.csv dir path
    
    :outputs: DataFrame - merged dataset
    
    """
    # load datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge datasets on common id and assign to df
    df = messages.merge(categories, how ='outer', on =['id'])
    return df



def clean_data(df):
    """
    This function is used to ensure data quality inside DataFrame
    
    :params: df - DataFrame
    :outputs: df - Cleaned DataFrame
    
    """
    # Split all categorical values from original DataFrame to a new DataFrame
    cate_cols = df['categories'].str.split(';', expand=True)
    
    # Extract first row of this DataFrame
    row = cate_cols.head(1)
    
    # Lambda function will slice each characters in each string, to the last two characters
    cate_colnames = row.applymap(lambda x: x[:-2]).iloc[0,:]
    # Rename old columns
    cate_cols.columns = cate_colnames
    
    # Iterate and keep the last character of the string
    for col in cate_cols:
        # Set each value to be the last character of the string and convert to numeric values
        cate_cols[col] = cate_cols[col].astype(str).str[-1]
        cate_cols[col] = cate_cols[col].astype(int)
    cate_cols['related'] = cate_cols['related'].replace(to_replace=2, value=1)
        
    # Drop the original categories column in df
    df.drop('categories', axis=1, inplace=True)
    
    # Concat and drop duplicates
    df = pd.concat([df, cate_cols], axis=1)
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename):
    """Stores df in a SQLite database."""
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('disaster_messages_tb', engine, index=False, if_exists='replace')  

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