'''
Given a list of stocks, return a list of stocks that meet the criteria
Use the efficient frontier to find the best portfolio
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()

from scipy.optimize import minimize


def get_data(tickers, start_date, end_date):
    '''
    Get stock data from yahoo finance
    '''
    data = pdr.get_data_yahoo(tickers, start=start_date, end=end_date)
    return data

def get_daily_returns(data):
    '''
    Calculate daily returns
    '''
    daily_returns = data['Adj Close'].pct_change()
    daily_returns = daily_returns.dropna()
    return daily_returns

def get_mean_daily_returns(daily_returns):
    '''
    Get mean daily returns
    '''
    mean_daily_returns = daily_returns.mean()
    return mean_daily_returns

def get_covariance_matrix(daily_returns):
    '''
    Get covariance matrix
    '''
    covariance_matrix = daily_returns.cov()
    return covariance_matrix

def get_portfolio_returns(weights, mean_daily_returns):
    '''
    Calculate portfolio returns
    '''
    portfolio_returns = np.sum(mean_daily_returns * weights)
    return portfolio_returns

def get_portfolio_volatility(weights, covariance_matrix):
    '''
    Calculate portfolio volatility
    '''
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
    return portfolio_volatility

def get_portfolio_sharpe_ratio(weights, mean_daily_returns, covariance_matrix, risk_free_rate):
    '''
    Calculate portfolio sharpe ratio
    '''
    portfolio_returns = get_portfolio_returns(weights, mean_daily_returns)
    portfolio_volatility = get_portfolio_volatility(weights, covariance_matrix)
    portfolio_sharpe_ratio = (portfolio_returns - risk_free_rate) / portfolio_volatility
    return portfolio_sharpe_ratio

def get_portfolio_weights(mean_daily_returns, covariance_matrix, risk_free_rate):
    '''
    Get portfolio weights
    '''
    num_assets = len(mean_daily_returns)
    args = (mean_daily_returns, covariance_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0, 1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = minimize(neg_sharpe, num_assets * [1. / num_assets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result['x']

def neg_sharpe(weights, mean_daily_returns, covariance_matrix, risk_free_rate):
    '''
    Negative sharpe ratio
    '''
    return get_portfolio_sharpe_ratio(weights, mean_daily_returns, covariance_matrix, risk_free_rate) * -1

def get_efficient_frontier(mean_daily_returns, covariance_matrix, risk_free_rate):
    '''
    Get efficient frontier
    '''
    efficient_portfolios = []
    num_assets = len(mean_daily_returns)
    args = (mean_daily_returns, covariance_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0, 1.0)
    bounds = tuple(bound for asset in range(num_assets))
    for portfolio_return in np.linspace(mean_daily_returns.min(), mean_daily_returns.max(), 100):
        constraints = ({'type': 'eq', 'fun': lambda x: get_portfolio_returns(x, mean_daily_returns) - portfolio_return},
                       {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        result = minimize(get_portfolio_volatility, num_assets * [1. / num_assets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
        efficient_portfolios.append(result['x'])
    return efficient_portfolios

def get_portfolio_performance(weights, mean_daily_returns, covariance_matrix, risk_free_rate):
    '''
    Get portfolio performance
    '''
    portfolio_returns = get_portfolio_returns(weights, mean_daily_returns)
    portfolio_volatility = get_portfolio_volatility(weights, covariance_matrix)
    portfolio_sharpe_ratio = get_portfolio_sharpe_ratio(weights, mean_daily_returns, covariance_matrix, risk_free_rate)
    return portfolio_returns, portfolio_volatility, portfolio_sharpe_ratio

def get_portfolio_performance_df(mean_daily_returns, covariance_matrix, risk_free_rate):
    '''
    Get portfolio performance dataframe
    '''
    portfolio_weights = get_portfolio_weights(mean_daily_returns, covariance_matrix, risk_free_rate)
    portfolio_returns, portfolio_volatility, portfolio_sharpe_ratio = get_portfolio_performance(portfolio_weights, mean_daily_returns, covariance_matrix, risk_free_rate)
    portfolio_performance_df = pd.DataFrame({'Returns': portfolio_returns, 'Volatility': portfolio_volatility, 'Sharpe Ratio': portfolio_sharpe_ratio}, index=['Portfolio'])
    return portfolio_performance_df

def get_efficient_frontier_df(mean_daily_returns, covariance_matrix, risk_free_rate):
    '''
    Get efficient frontier dataframe
    '''
    efficient_portfolios = get_efficient_frontier(mean_daily_returns, covariance_matrix, risk_free_rate)
    efficient_frontier_df = pd.DataFrame(columns=['Returns', 'Volatility', 'Sharpe Ratio'])
    for weights in efficient_portfolios:
        portfolio_returns, portfolio_volatility, portfolio_sharpe_ratio = get_portfolio_performance(weights, mean_daily_returns, covariance_matrix, risk_free_rate)
        efficient_frontier_df = efficient_frontier_df.append({'Returns': portfolio_returns, 'Volatility': portfolio_volatility, 'Sharpe Ratio': portfolio_sharpe_ratio}, ignore_index=True)
    return efficient_frontier_df


def plot_efficient_frontier_with_portfolio(mean_daily_returns, covariance_matrix, risk_free_rate):
    '''
    Plot efficient frontier with portfolio
    '''
    portfolio_performance_df = get_portfolio_performance_df(mean_daily_returns, covariance_matrix, risk_free_rate)
    efficient_frontier_df = get_efficient_frontier_df(mean_daily_returns, covariance_matrix, risk_free_rate)
    ax = efficient_frontier_df.plot.scatter(x='Volatility', y='Returns', c='Sharpe Ratio', cmap='viridis', edgecolors='k', figsize=(10, 8), grid=True)
    portfolio_performance_df.plot.scatter(x='Volatility', y='Returns', c='red', marker='X', s=200, ax=ax)
    plt.xlabel('Volatility (Std. Deviation)')
    plt.ylabel('Expected Returns')
    plt.title('Efficient Frontier')
    plt.show()

def get_mean_daily_returns_and_covariance_matrix():
    '''
    Get mean daily returns and covariance matrix
    '''
    df = get_data()
    mean_daily_returns = df.mean()
    covariance_matrix = df.cov()
    return mean_daily_returns, covariance_matrix

def main():
    '''
    Main function
    '''
    risk_free_rate = 0.0178
    mean_daily_returns, covariance_matrix = get_mean_daily_returns_and_covariance_matrix()
    plot_efficient_frontier_with_portfolio(mean_daily_returns, covariance_matrix, risk_free_rate)