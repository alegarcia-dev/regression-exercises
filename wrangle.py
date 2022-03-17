################################################################################
#
#
#
#       wrangle.py
#
#       Description: Functions needed for wrangling the zillow dataset.
#
#       Variables:
#
#           _zillow_file
#           _zillow_db
#
#       Functions:
#
#           get_zillow_data(use_cache)
#           _get_zillow_sql()
#           prepare_zillow_data(df)
#
#
#
################################################################################

import numpy as np
import pandas as pd

from get_db_url import get_db_url

################################################################################

_zillow_file = 'zillow.csv'
_zillow_db = 'zillow'

################################################################################

def get_zillow_data(use_cache: bool = True) -> pd.core.frame.DataFrame:
    '''
        Return a dataframe containing data from the zillow properties dataset.

        If a zillow.csv file containing the data does not already
        exist the data will be cached in that file inside the current
        working directory. Otherwise, the data will be read from the
        .csv file.

        Parameters
        ----------
        use_cache: bool, default True
            If True the dataset will be retrieved from a csv file if one
            exists, otherwise, it will be retrieved from the MySQL database. 
            If False the dataset will be retrieved from the MySQL database
            even if the csv file exists.

        Returns
        -------
        DataFrame: A Pandas DataFrame containing the data from the zillow
            dataset is returned.
    '''

    # If the file is cached, read from the .csv file
    if os.path.exists(_zillow_file) and use_cache:
        return pd.read_csv(_zillow_file)
    
    # Otherwise read from the mysql database
    else:
        df = pd.read_sql(_get_zillow_sql(), get_db_url(_zillow_db))
        df.to_csv(_zillow_file, index = False)
        return df

################################################################################
    
def _get_zillow_sql() -> str:
    '''
        Returns the SQL code required to retrieve the zillow dataset
        from the MySQL database.
    '''

    return '''
        SELECT
            bedroomcnt,
            bathroomcnt,
            calculatedfinishedsquarefeet,
            taxvaluedollarcnt,
            yearbuilt,
            taxamount,
            fips,
            propertylandusedesc
        FROM properties_2017
        JOIN propertylandusetype
            ON propertylandusetype.propertylandusetypeid = properties_2017.propertylandusetypeid
            AND propertylandusetype.propertylandusedesc = 'Single Family Residential';
    '''

################################################################################

def prepare_zillow_data(df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    '''
        Returns a prepared zillow dataset with all missing values handled.
        
        Parameters
        ----------
        df: DataFrame
            A pandas dataframe containing the unprepared zillow dataset.
        
        Returns
        -------
        DataFrame: A pandas dataframe containing the prepared zillow dataset.
    '''
    
    # Remove rows with two or more missing values.
    rows_with_one_or_less_missing_values = df.isnull().sum(axis = 1) < 2
    df = df[rows_with_one_or_less_missing_values]

    # Remove rows missing square footage
    rows_not_missing_square_feet = df.calculatedfinishedsquarefeet.notnull()
    df = df[rows_not_missing_square_feet]

    # Remove rows missing taxvaluedollarcnt
    rows_not_missing_taxvalue = df.taxvaluedollarcnt.notnull()
    df = df[rows_not_missing_taxvalue]
    
    # Fill in yearbuilt column with the mode and cast to int
    df.yearbuilt.fillna(df.yearbuilt.mode()[0], inplace = True)
    df.yearbuilt = df.yearbuilt.astype('int')
    
    # Fill in the taxamount column with the mean
    df.taxamount.fillna(df.taxamount.mean(), inplace = True)

    return df

################################################################################

def wrangle_zillow() -> pd.core.frame.DataFrame:
    '''
        Returns the acquired and prepared zillow dataset.
        
        Returns
        -------
        DataFrame: A pandas dataframe containing the prepared zillow dataset.
    '''
    
    return prepare_zillow_data(get_zillow_data())