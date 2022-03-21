################################################################################
#
#
#
#       explore.py
#
#       Description: This file contains functions used in the explore exercises
#           of the regression module for the Codeup data science program.
#
#       Fields:
#
#           None
#
#       Functions:
#
#           plot_variable_pairs(df)
#           months_to_years(df)
#           plot_categorical_and_continuous_vars(df, categorical_cols, continuous_cols)
#
#
################################################################################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

################################################################################

def plot_variable_pairs(df: pd.core.frame.DataFrame) -> None:
    '''
        Display a plot of all pairwise relationships in the dataframe
        provided with a regression line in each plot.
    
        Parameters
        ----------
        df: DataFrame
            A pandas dataframe containing the data to be plotted.
    '''

    sns.pairplot(df.drop(columns = 'customer_id'), corner = True, kind = 'reg')
    plt.show()

################################################################################

def months_to_years(df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    '''
        Returns a new dataframe with a new column added for the years of 
        tenure a customer has.
    
        Parameters
        ----------
        df: DataFrame
            A pandas dataframe containing the Telco dataset.
    
        Returns
        -------
        DataFrame: A pandas dataframe containing the Telco dataset with 
            a new column for tenure_years.
    '''

    df = df.copy()
    df['tenure_years'] = round(df.tenure / 12, 0).astype('int')
    return df

################################################################################

def plot_categorical_and_continuous_vars(
    df: pd.core.frame.DataFrame,
    categorical_cols: list[str],
    continuous_cols: list[str]
) -> None:
    '''
        Plot a boxplot, barplot, and histplot for each combination of continous 
        and categorical column in the dataframe provided.
    
        Parameters
        ----------
        df: DataFrame
            A pandas dataframe containing the data to be plotted.

        categorical_cols: list[str]
            A list of the categorical columns to plot.

        continuous_cols: list[str]
            A list of the continuous columns to plot.
    '''

    for con in continuous_cols:
        for cat in categorical_cols:
            fig = plt.figure(figsize = (14, 4))
            fig.suptitle(f'{con} v. {cat}')

            plt.subplot(1, 3, 1)
            sns.boxplot(data = df, x = cat, y = con)
            plt.axhline(df[con].mean())

            plt.subplot(1, 3, 2)
            sns.barplot(data = df, x = cat, y = con)
            plt.axhline(df[con].mean())

            plt.subplot(1, 3, 3)
            sns.histplot(data = df, x = con, bins = 10, hue = cat)
            plt.show()