################################################################################
#
#
#
#       evaluate.py
#
#       Description: This file contains functions used for evaluating regression
#           models.
#
#       Variables:
#
#           None
#
#       Functions:
#
#           plot_residuals(actual, predictions)
#           regression_errors(actual, predictions, print_results = True)
#           baseline_mean_errors(actual, baseline, print_results = True)
#           better_than_baseline(actual, predictions)
#           _SSE(actual, predictions)
#           _ESS(actual, predictions)
#           _TSS(actual, predictions)
#           _MSE(actual, predictions)
#           _RMSE(actual, predictions)
#
#
################################################################################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error, explained_variance_score

################################################################################

def plot_residuals(actual, predictions):
    '''
        Create a residual plot using the predictions from a regression model 
        and the actual values.
    
        Parameters
        ----------
        actual: Series
            A pandas series containing the actual values from a dataset.
    
        predictions: Array
            A numpy array containing the predictions from a regression model.
    '''

    residuals = actual - predictions
    plt.axhline(0, ls=':')
    plt.scatter(actual, residuals)
    plt.xlabel('Actual')
    plt.ylabel('Residual')
    plt.title('Residuals for yhat')

################################################################################

def regression_errors(actual, predictions, print_results: bool = True) -> pd.core.series.Series:
    '''
        Print or return the error metrics for a regression model (SSE, ESS,
        TSS, MSE, and RMSE).
    
        Parameters
        ----------
        actual: Series
            A pandas series containing the actual values from a dataset.

        predictions: Array
            A numpy array containing the predictions from a regression model.

        print_results: bool, default True
            If True the metric scores are printed to the console, if False 
            the metric scores are returned.
    
        Returns
        -------
        Series | None: If print_results is False returns a pandas series of 
            floats containing the metric scores for the baseline model.
    '''

    if print_results:
        print(f'''
            sum of squared errors (SSE):     {_SSE(actual, predictions)}
            explained sum of squares (ESS):  {_ESS(actual, predictions)}
            total sum of squares (TSS):      {_TSS(actual, predictions)}
            mean squared error (MSE):        {_MSE(actual, predictions)}
            root mean squared error (RMSE):  {_RMSE(actual, predictions)}
        ''')
    else:
        return pd.Series({
            'SSE' : _SSE(actual, predictions),
            'ESS' : _ESS(actual, predictions),
            'TSS' : _TSS(actual, predictions),
            'MSE' : _MSE(actual, predictions),
            'RMSE' : _RMSE(actual, predictions)
        })

################################################################################

def baseline_mean_errors(actual, baseline, print_results: bool = True) -> pd.core.series.Series:
    '''
        Print or return the baseline error metrics for a regression model (SSE, 
        MSE, and RMSE).
    
        Parameters
        ----------
        actual: Series
            A pandas series containing the actual values from a dataset.

        predictions: Array
            A numpy array containing the predictions from a regression model.

        print_results: bool, default True
            If True the metric scores are printed to the console, if False 
            the metric scores are returned.
    
        Returns
        -------
        Series | None: If print_results is False returns a pandas series of 
            floats containing the metric scores for the baseline model.
    '''

    if print_results:
        print(f'''
            Baseline sum of squared errors (SSE):     {_SSE(actual, baseline)}
            Baseline mean squared error (MSE):        {_MSE(actual, baseline)}
            Baseline root mean squared error (RMSE):  {_RMSE(actual, baseline)}
        ''')
    else:
        return pd.Series({
            'SSE' : _SSE(actual, baseline),
            'MSE' : _MSE(actual, baseline),
            'RMSE' : _RMSE(actual, baseline)
        })

################################################################################

def better_than_baseline(actual, predictions) -> bool:
    '''
        Returns True if the model's predictions are better than the baseline's 
        predictions using the root mean squared error score.
    
        Parameters
        ----------
        actual: Series
            A pandas series containing the actual values from a dataset.

        predictions: Array
            A numpy array containing the predictions from a regression model.
    
        Returns
        -------
        bool: Whether or not the model performs better than the baseline.
    '''

    baseline = _RMSE(actual, pd.Series([actual.mean()] * len(actual)))
    return _RMSE(actual, predictions) < baseline

################################################################################

def _SSE(actual, predictions) -> float:
    '''
        Returns the sum of squared errors score for a regression model.
    
        Parameters
        ----------
        actual: Series
            A pandas series containing the actual values from a dataset.

        predictions: Array
            A numpy array containing the predictions from a regression model.
    
        Returns
        -------
        float: The sum of squared errors for a regression model.
    '''

    return mean_squared_error(actual, predictions) * len(actual)

################################################################################

def _ESS(actual, predictions) -> float:
    '''
        Returns the explained sum of squares score for a regression model.
    
        Parameters
        ----------
        actual: Series
            A pandas series containing the actual values from a dataset.

        predictions: Array
            A numpy array containing the predictions from a regression model.
    
        Returns
        -------
        float: The explained sum of squares for a regression model.
    '''

    return sum((predictions - actual.mean()) ** 2)

################################################################################

def _TSS(actual, predictions) -> float:
    '''
        Returns the total sum of squares score for a regression model.
    
        Parameters
        ----------
        actual: Series
            A pandas series containing the actual values from a dataset.

        predictions: Array
            A numpy array containing the predictions from a regression model.
    
        Returns
        -------
        float: The total sum of squares score for a regression model.
    '''

    return _SSE(actual, predictions) + _ESS(actual, predictions)

################################################################################

def _MSE(actual, predictions) -> float:
    '''
        Returns the mean squared error score for a regression model.
    
        Parameters
        ----------
        actual: Series
            A pandas series containing the actual values from a dataset.

        predictions: Array
            A numpy array containing the predictions from a regression model.
    
        Returns
        -------
        float: The mean squared error score for a regression model.
    '''

    return mean_squared_error(actual, predictions)

################################################################################

def _RMSE(actual, predictions) -> float:
    '''
        Returns the root mean squared error score for a regression model.
    
        Parameters
        ----------
        actual: Series
            A pandas series containing the actual values from a dataset.

        predictions: Array
            A numpy array containing the predictions from a regression model.
    
        Returns
        -------
        float: The root mean squared error score for a regression model.
    '''

    return mean_squared_error(actual, predictions, squared = False)