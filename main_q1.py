import pandas as pd
from asset import asset
from data import data_handler
from strategy import strategy
from portfolio import portfolio

# turn off the pandas warning
pd.options.mode.chained_assignment = None

# read data from excel
price_data = data_handler.DataHandler.load_excel_data('./data/Coding_Proj_Data.xls', sheet_name='Price'
                                                      ).iloc[:, 0:2]
dividend_data = data_handler.DataHandler.load_excel_data('./data/Coding_Proj_Data.xls', sheet_name='Dividend'
                                                         ).iloc[:, 0:5].rename(columns={'Amount': 'Dividends'})
curve_rate_data = data_handler.DataHandler.load_excel_data('./data/Coding_Proj_Data.xls', sheet_name='Interest Rate')
# set the first row of curve_rate_data the tenor and make it column name
curve_rate_data.columns = curve_rate_data.iloc[0]
curve_rate_data = curve_rate_data.iloc[1:, :].reset_index(drop=True)
# curve data issue on Tenor(D) 7.0 and 60.0 after 2022-01-01, fill with NA, interpolation will be done later
curve_rate_data.loc['2022-01-01':, [7.0, 60.0]] = None
# curve data issue on Tenor(D) 1 and 360 after 2023-07-01, send warning
print('Suspect Curve data issue on Tenor(D) 1 and 360 after 2023-07-01!')

# clean data
price_data = data_handler.DataHandler.clean_time_series(price_data, 'Date')
dividend_data = data_handler.DataHandler.clean_time_series(dividend_data, 'ExDate')
curve_rate_data = data_handler.DataHandler.clean_time_series(curve_rate_data, 'Date', how_drop_na='all')
cash_rate_data = curve_rate_data.loc[:, 1].to_frame('cash_rate')

# check data
data_handler.DataHandler.print_missing_data_pct_by_col(price_data)
data_handler.DataHandler.print_missing_data_pct_by_col(dividend_data)

# create ETF Asset
# assume 3 cents per share transaction cost
spy_etf = asset.ETF(name='SPY', master_ticker='SPY', price=price_data, dividend=dividend_data, transaction_cost=0.03)
spy_etf.total_return_calc()
spy_etf.forward_return_generate()

# create Cash Asset
cash = asset.Cash(name='Cash', master_ticker='CASH', cash_rate=cash_rate_data, currency='USD')
# convention could be ACT/365 or ACT/252, I choose 252 here
# because I remove non-business date in previous steps to make data consistent, there are ~252 business days in a year
# is an appropriate assumption
cash.total_return_calc(convention='ACT/252')

# create moving average cross strategy
strategy_data = spy_etf.total_return_index.to_frame()
strategy_data['master_ticker'] = 'SPY'
ma_cross_strategy = strategy.MovingAverageCrossStrategy(name='Moving Average Cross Strategy 50D vs 200D',
                                                        data=strategy_data, short_window=50, long_window=200,
                                                        price_col_name='total_return_index', start_date='2005-01-03',
                                                        end_date='2024-01-23')
ma_cross_strategy.signal_generation()
# ma_cross_strategy.signal_plot('SPY') # plot the signal

# experiment with RSI strategy function
# rsi_strategy = strategy.RSI(name='RSI Strategy', data=strategy_data, window=14, price_col_name='total_return_index',
#                             start_date='2005-01-03', end_date='2024-01-23')
# rsi_strategy.signal_generation()
# rsi_strategy.signal_plot('SPY')

# create portfolio
portfolio_start_date = max(spy_etf.total_return_index.index.min(), cash.cash_rate.index.min(),
                           ma_cross_strategy.signal_history.index.min(),
                           pd.to_datetime('2005-06-30')).strftime('%Y-%m-%d')
portfolio_end_date = min(spy_etf.total_return_index.index.max(), cash.cash_rate.index.max(),
                         pd.to_datetime('2023-12-31')).strftime('%Y-%m-%d')

spy_trend_following_portfolio = portfolio.MultiAssetPortfolio(name='SPX Trend Following', asset_list=[spy_etf, cash],
                                                              start_date=portfolio_start_date,
                                                              end_date=portfolio_end_date,
                                                              initial_capital=1000000, rebalance_freq=None,
                                                              signal_data=ma_cross_strategy.signal_history,
                                                              benchmark=spy_etf.total_return_index.to_frame('SPY'))
spy_trend_following_portfolio.backtesting(is_generate_output=True)
print('Done!')
