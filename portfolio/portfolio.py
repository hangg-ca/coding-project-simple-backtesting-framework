import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from data import data_handler
import tqdm
# import cvxpy as cp

########################################################################################################################
# This code is only for demonstration purpose and not for real trading. Any changes are not suggested by the creator of
# this code. The creator of this code is not responsible for any loss caused by using this code.
# The user should use this code at their own risk.
# Creator: Hangg https://github.com/hangg-ca
########################################################################################################################

class BasePortfolio(object):
    '''
    Portfolio class: Portfolio class as parent class for all portfolios
    '''

    def __init__(self, **kwargs):
        self.name = kwargs['name']
        self.asset_list = kwargs['asset_list']

        # type check on asset_list to be list
        if not isinstance(self.asset_list, list):
            raise TypeError("Asset list must be list")

    # void method
    # def performance_analysis(self):
    #     raise NotImplementedError("Should have implemented performance_analysis()")
    #
    # # void method
    # def risk_analysis(self):
    #     raise NotImplementedError("Should have implemented risk_analysis()")
    #
    # # void method
    # def portfolio_construction(self):
    #     raise NotImplementedError("Should have implemented portfolio_construction()")

    # # void method
    # def portfolio_rebalance(self):
    #     raise NotImplementedError("Should have implemented portfolio_rebalance()")


class MultiAssetPortfolio(BasePortfolio):
    """
    MultiAssetPortfolio class: MultiAssetPortfolio class as child class of Portfolio class
    """

    def __init__(self, **kwargs):
        BasePortfolio.__init__(self, **kwargs)
        self.start_date = kwargs['start_date']
        self.end_date = kwargs['end_date']
        self.initial_capital = kwargs['initial_capital']
        self.rebalance_freq = kwargs['rebalance_freq']
        if 'signal_data' in kwargs:
            self.signal_data = kwargs['signal_data']
        else:
            self.signal_data = None  # No signal data, typically for static portfolio only
        # combine all asset price data into one dataframe
        self.price_data = pd.DataFrame()
        self.combined_total_return = pd.DataFrame()
        self.combined_dividend = pd.DataFrame()
        self.transaction_cost_dict = {}
        self.contract_unit_dict = {}
        for asset in self.asset_list:
            self.price_data = pd.concat([self.price_data, asset.price.rename(
                columns={'Price': asset.master_ticker})], axis=1)
            self.combined_total_return = pd.concat([self.combined_total_return, asset.total_return_index.to_frame(
                asset.master_ticker)], axis=1)
            if hasattr(asset, 'dividend'):
                self.combined_dividend = pd.concat([self.combined_dividend, asset.dividend.rename(
                    columns={'Dividends': asset.master_ticker})], axis=1)
            self.transaction_cost_dict[asset.master_ticker] = asset.transaction_cost
            if hasattr(asset, 'contract_unit'):
                self.contract_unit_dict[asset.master_ticker] = asset.contract_unit
            else:
                self.contract_unit_dict[asset.master_ticker] = 1
        # slice price data and total return data by start and end date
        self.price_data = self.price_data.sort_index().loc[self.start_date:self.end_date, :]
        # if duplicated master ticker in price data, combine them to one
        self.price_data = self.price_data.groupby(self.price_data.columns, axis=1).mean()
        self.combined_total_return = self.combined_total_return.sort_index().loc[self.start_date:self.end_date, :]
        self.combined_total_return = self.combined_total_return.groupby(self.combined_total_return.columns, axis=1
                                                                        ).mean()
        self.combined_dividend = self.combined_dividend.sort_index().loc[self.start_date:self.end_date, :]
        # initialize holdings
        if 'holdings' in kwargs:
            self.holdings = kwargs['holdings']

        else:
            self.holdings = pd.DataFrame(index=self.combined_total_return.index,
                                         columns=self.combined_total_return.columns)
            self.holdings.fillna(0, inplace=True)
        if 'holdings_by_weight' in kwargs:
            self.holdings_by_weight = kwargs['holdings_by_weight']
        else:
            self.holdings_by_weight = None
        if 'benchmark' in kwargs:
            self.benchmark = kwargs['benchmark']
            # check benchmark data is pandas dataframe
            if not isinstance(self.benchmark, pd.DataFrame):
                raise TypeError("Benchmark must be pandas dataframe")
            self.benchmark_name = self.benchmark.columns[0]
        else:
            self.benchmark = pd.DataFrame(index=self.combined_total_return.index,
                                          columns=['total_return_index']).fillna(0)
        # initialize portfolio account
        self.portfolio_account = pd.DataFrame(index=self.combined_total_return.index,
                                              columns=['NAV', 'Income', 'Available_Cash'])
        self.portfolio_account.loc[self.start_date, 'NAV'] = self.initial_capital
        self.portfolio_account.loc[self.start_date, 'Available_Cash'] = self.initial_capital
        self.portfolio_account.loc[self.start_date, 'Unavailable_Cash'] = 0
        self.portfolio_account.loc[self.start_date, 'Income'] = 0
        # initialize trade records
        self.trade_records = pd.DataFrame(columns=['Trade Date', 'Master_Ticker', 'Amount', 'Price', 'Transaction_Cost',
                                                   'Side', 'Trade_Rational'])

    def trend_following_portfolio_construction(self, rebalance_date):
        """
        if signal is 1, invest all money in SPY;
        if signal is -1, sell the existing position;
        if nothing happened, park the money in cash, earning money market rate. No short or leverage is allowed.

        """
        # get the signal on rebalance date
        signal = self.signal_data.loc[rebalance_date, 'signal']
        # get the price on rebalance date
        price = self.price_data.loc[rebalance_date, :]
        # get the holdings on rebalance date
        holdings = self.holdings.loc[rebalance_date, :]
        # get the portfolio account on rebalance date
        portfolio_account = self.portfolio_account.loc[rebalance_date, :]
        # get the total return on rebalance date
        total_return = self.combined_total_return.loc[rebalance_date, :]
        available_cash = portfolio_account['Available_Cash']
        # if signal is 1, invest all money in SPY;
        if signal == 1:
            # if already invested in SPY, and not receive dividend, do nothing
            if (holdings['SPY'] != 0) & (rebalance_date not in self.combined_dividend['PayDate'].values):
                pass
            # if not invested in SPY, or receive dividend, buy SPY
            else:
                # print('Buy SPY date: ', rebalance_date)
                # sell all existing position except SPY
                for master_ticker in holdings[holdings.index != 'SPY'].index:
                    if holdings[master_ticker] != 0:
                        if master_ticker not in ['CASH']:  # if the sell is not cash
                            # record trade records
                            sell_amount = holdings[master_ticker]
                            temp_trade_records = pd.DataFrame([[rebalance_date, master_ticker, sell_amount,
                                                                price[master_ticker],
                                                                self.transaction_cost_dict[master_ticker] * np.abs(
                                                                    sell_amount), 'Sell', 'Signal Driven']],
                                                              columns=self.trade_records.columns)
                            self.trade_records = pd.concat([self.trade_records, temp_trade_records], axis=0)
                            # update holdings
                            holdings[master_ticker] = 0
                            # transaction cost
                            available_cash += (price[master_ticker] - self.transaction_cost_dict[master_ticker]
                                               ) * holdings[master_ticker]
                        else:
                            holdings[master_ticker] = 0
                # buy SPY
                # record trade records
                buy_amount = available_cash / (price['SPY'] + self.transaction_cost_dict['SPY'])
                buy_amount = np.floor(buy_amount)
                if rebalance_date in self.combined_dividend['PayDate'].values:
                    temp_trade_records = pd.DataFrame([[rebalance_date, 'SPY', buy_amount, price['SPY'],
                                                        self.transaction_cost_dict['SPY'] * np.abs(buy_amount),
                                                        'Buy', 'Dividend Reinvestment']],
                                                      columns=self.trade_records.columns)
                else:
                    temp_trade_records = pd.DataFrame([[rebalance_date, 'SPY', buy_amount, price['SPY'],
                                                        self.transaction_cost_dict['SPY'] * np.abs(buy_amount), 'Buy',
                                                        'Signal Driven']],
                                                      columns=self.trade_records.columns)
                self.trade_records = pd.concat([self.trade_records, temp_trade_records], axis=0)
                # update holdings
                holdings['SPY'] += buy_amount
                available_cash -= (price['SPY'] + self.transaction_cost_dict['SPY']) * buy_amount
                holdings['CASH'] = available_cash

        # if signal is -1, sell the existing SPY position;
        elif signal == -1:
            # if still invested in SPY, sell all existing position
            if holdings['SPY'] != 0:
                # print('Sell SPY date: ', rebalance_date)
                # record trade records
                sell_amount = holdings['SPY']
                temp_trade_records = pd.DataFrame([[rebalance_date, 'SPY', sell_amount, price['SPY'],
                                                    self.transaction_cost_dict['SPY'] * np.abs(sell_amount), 'Sell',
                                                    'Signal Driven']],
                                                  columns=self.trade_records.columns)
                self.trade_records = pd.concat([self.trade_records, temp_trade_records], axis=0)
                # update holdings
                holdings['SPY'] = 0
                # transaction cost
                available_cash += (price['SPY'] - self.transaction_cost_dict['SPY']) * sell_amount
                holdings['CASH'] = available_cash
            else:
                pass
            # update portfolio account
        # update nav
        portfolio_account['NAV'] = (holdings * price).sum() + portfolio_account['Unavailable_Cash']
        portfolio_account['Available_Cash'] = available_cash

        # update holdings
        self.holdings.loc[rebalance_date, :] = holdings
        # update portfolio account
        self.portfolio_account.loc[rebalance_date, :] = portfolio_account
        pass

    def collar_portfolio_construction(self, rebalance_date):
        """
        if signal is 1, means long the position
        if signal is -1, means short the position
        a)	Long SPY
        b)	Buy SPY 95% strike OTM put.
        The put notional is 100% of the long position to provide protection.
        The maturities of the contracts are evenly spread across 3, 6, 9 and 12-month.
        c)	Sell SPY 105% strike OTM call.
        The call notional is 100% of the long position to generate income. The maturity of the contracts is 1 month.
        The construction function should find optimal number of contracts and underlyings
        to minimize the cash left in the portfolio.
        """

        def cost_function(x):
            trading_cost = ((x['New'] - x['Cur']).abs() * x['Transaction_Cost']).sum()
            used_cash_for_trade = ((x['New'] - x['Cur']) * x['Price'] * x['Contract_Unit']).sum()
            return trading_cost + used_cash_for_trade, trading_cost, used_cash_for_trade
        # get the signal on rebalance date
        signal = self.signal_data.loc[rebalance_date, ['signal', 'master_ticker']].copy()
        # get the price on rebalance date
        price = self.price_data.loc[rebalance_date, :].copy()
        # get the holdings on rebalance date
        holdings = self.holdings.loc[rebalance_date, :].copy()
        # get the portfolio account on rebalance date
        portfolio_account = self.portfolio_account.loc[rebalance_date, :].copy()
        # get the total return on rebalance date
        total_return = self.combined_total_return.loc[rebalance_date, :].copy()
        available_cash = portfolio_account['Available_Cash']
        # if it just month end, only need to roll call option
        # if it is not month end, need to roll call option and buy put option, indicate a full portfolio rebalance
        # check if any of the put option holding has signal is 0
        holdings_put = holdings[(holdings.index.str.contains('OPTION_PUT')) & (holdings != 0)]
        no_holding_option = signal.loc[signal['signal'] == 0, 'master_ticker']

        long_asset_ticker = signal.loc[signal['signal'] == 1, 'master_ticker'].tolist()
        write_asset_ticker = signal.loc[signal['signal'] == -1, 'master_ticker'].tolist()
        equity_asset_ticker = [i for i in long_asset_ticker if i in long_asset_ticker if 'OPTION' not in i][0]
        long_option_ticker = [i for i in long_asset_ticker if i in long_asset_ticker if 'OPTION' in i]

        # estimate the cost of roll call option
        holding_call = holdings[(holdings.index.str.contains('OPTION_CALL')) & (holdings != 0)]
        roll_cost_est = 0
        if not holding_call.empty:
            new_call = pd.Series(index=write_asset_ticker, data=0)
            combined_position = pd.concat([holding_call.to_frame('Cur'), new_call.to_frame('New')], axis=1)
            combined_position = combined_position.loc[
                ~((combined_position['Cur'] == 0) & (combined_position['New'].isna()))]
            combined_position = combined_position[~combined_position.index.isin(['CASH'])]
            combined_position['Price'] = price[combined_position.index]
            combined_position['Contract_Unit'] = combined_position.index.map(self.contract_unit_dict)
            combined_position['Transaction_Cost'] = combined_position.index.map(self.transaction_cost_dict)
            combined_position.loc[write_asset_ticker, 'New'] = -(holdings.loc[equity_asset_ticker] /
                                                                 len(write_asset_ticker) /
                                                                 combined_position.loc[write_asset_ticker,
                                                                 'Contract_Unit'])
            combined_position.fillna(0, inplace=True)
            roll_cost_est = cost_function(combined_position)[0]

        if holdings_put.empty:
            rebalance_mode = 'full'
        elif holdings_put.index.isin(no_holding_option).any():
            rebalance_mode = 'full'
        elif roll_cost_est > available_cash:
            rebalance_mode = 'full'
        else:
            rebalance_mode = 'roll'

        if rebalance_mode == 'full':
            # based on available cash, calculate the number of put and SPY option to buy, call to write

            # option 1: use for loop to search for max number of SPY that could potentially hold
            # another option is to use cvxpy to solve the MIP optimization problem to maximize the number of SPY
            potential_buy_power = portfolio_account['NAV'] - portfolio_account['Unavailable_Cash']
            initial_underlying_holding = potential_buy_power / price[equity_asset_ticker]
            # equity rounding unit is the max of all the contract unit and could be divided by number of options
            new_equity_rounding_unit = max([self.contract_unit_dict[equity_asset_ticker]] +
                                           [self.contract_unit_dict[i] * len(long_option_ticker) for i in
                                            long_option_ticker] +
                                           [self.contract_unit_dict[i] * len(write_asset_ticker) for i in
                                            write_asset_ticker])

            initial_underlying_holding = np.floor(initial_underlying_holding / new_equity_rounding_unit
                                                  ) * new_equity_rounding_unit
            new_holdings = pd.Series(index=long_asset_ticker + write_asset_ticker, data=0)
            combined_position = pd.concat([holdings.to_frame('Cur'), new_holdings.to_frame('New')], axis=1)
            combined_position = combined_position.loc[
                ~((combined_position['Cur'] == 0) & (combined_position['New'].isna()))]
            combined_position = combined_position[~combined_position.index.isin(['CASH'])]
            combined_position['Price'] = price[combined_position.index]
            combined_position['Contract_Unit'] = combined_position.index.map(self.contract_unit_dict)
            combined_position['Transaction_Cost'] = combined_position.index.map(self.transaction_cost_dict)

            combined_position.loc[equity_asset_ticker, 'New'] = initial_underlying_holding
            combined_position.loc[write_asset_ticker, 'New'] = -(initial_underlying_holding / len(write_asset_ticker) /
                                                                 combined_position.loc[write_asset_ticker,
                                                                 'Contract_Unit'])
            combined_position.loc[long_option_ticker, 'New'] = (initial_underlying_holding / len(long_option_ticker) /
                                                                combined_position.loc[long_option_ticker,
                                                                'Contract_Unit'])
            combined_position.fillna(0, inplace=True)

            number_iteration = 0
            while cost_function(combined_position)[0] > available_cash:
                if number_iteration > 100:
                    raise ValueError('Cannot find the optimal solution')
                initial_underlying_holding -= new_equity_rounding_unit
                combined_position.loc[equity_asset_ticker, 'New'] = initial_underlying_holding
                combined_position.loc[write_asset_ticker, 'New'] = -(
                        initial_underlying_holding / len(write_asset_ticker) /
                        combined_position.loc[write_asset_ticker,
                        'Contract_Unit'])
                combined_position.loc[long_option_ticker, 'New'] = (
                        initial_underlying_holding / len(long_option_ticker) /
                        combined_position.loc[long_option_ticker,
                        'Contract_Unit'])
                number_iteration += 1
            # record trade records
            for master_ticker in combined_position.index:
                if combined_position.loc[master_ticker, 'New'] != combined_position.loc[master_ticker, 'Cur']:
                    if combined_position.loc[master_ticker, 'New'] > combined_position.loc[master_ticker, 'Cur']:
                        side = 'Buy'
                    else:
                        side = 'Sell'
                    temp_trade_records = pd.DataFrame([[rebalance_date, master_ticker,
                                                        combined_position.loc[master_ticker, 'New'] -
                                                        combined_position.loc[master_ticker, 'Cur'],
                                                        combined_position.loc[master_ticker, 'Price'],
                                                        combined_position.loc[master_ticker, 'Transaction_Cost'] *
                                                        np.abs(combined_position.loc[master_ticker, 'New'] -
                                                               combined_position.loc[master_ticker, 'Cur']),
                                                        side, 'Rebalance']],
                                                      columns=self.trade_records.columns)
                    self.trade_records = pd.concat([self.trade_records, temp_trade_records], axis=0)
            # update holdings
            # write all holdings to 0 then update the new holdings
            holdings = pd.Series(index=holdings.index, data=0)
            holdings.loc[combined_position.index] = combined_position['New']
            holdings['CASH'] = available_cash - cost_function(combined_position)[0]
            # update portfolio account
            portfolio_account['NAV'] = (combined_position['New'] * combined_position['Price'] * combined_position[
                'Contract_Unit']).sum() + portfolio_account['Unavailable_Cash'] + holdings['CASH']
            portfolio_account['Available_Cash'] = holdings['CASH']
            # update holdings
            self.holdings.loc[rebalance_date, holdings.index] = holdings
            # update portfolio account
            self.portfolio_account.loc[rebalance_date, :] = portfolio_account
        elif rebalance_mode == 'roll':
            # record trade records
            for master_ticker in combined_position.index:
                if combined_position.loc[master_ticker, 'New'] != combined_position.loc[master_ticker, 'Cur']:
                    if combined_position.loc[master_ticker, 'New'] > combined_position.loc[master_ticker, 'Cur']:
                        side = 'Buy'
                    else:
                        side = 'Sell'
                    temp_trade_records = pd.DataFrame([[rebalance_date, master_ticker,
                                                        combined_position.loc[master_ticker, 'New'] -
                                                        combined_position.loc[master_ticker, 'Cur'],
                                                        combined_position.loc[master_ticker, 'Price'],
                                                        combined_position.loc[master_ticker, 'Transaction_Cost'] *
                                                        np.abs(combined_position.loc[master_ticker, 'New'] -
                                                               combined_position.loc[master_ticker, 'Cur']),
                                                        side, 'Roll']],
                                                      columns=self.trade_records.columns)
                    self.trade_records = pd.concat([self.trade_records, temp_trade_records], axis=0)
            # update holdings
            # write all holdings to 0 then update the new holdings
            holdings.loc[combined_position.index] = combined_position['New']
            holdings['CASH'] = available_cash - cost_function(combined_position)[0]
            # update portfolio account
            portfolio_account['NAV'] = (holdings * price[holdings.index] * holdings.index.map(self.contract_unit_dict)
                                        ).sum() + portfolio_account['Unavailable_Cash']
            self.holdings.loc[rebalance_date].to_frame().merge(price.to_frame(), left_index=True, right_index=True)
            portfolio_account['Available_Cash'] = holdings['CASH']
            # update holdings
            self.holdings.loc[rebalance_date, holdings.index] = holdings
            # update portfolio account
            self.portfolio_account.loc[rebalance_date, :] = portfolio_account
        else:
            # regular day without any action
            pass

    def portfolio_prerebalance_accounting(self, rebalance_date):
        """
        the function to calculate the overnight return and dividend income
        meanwhile update the portfolio account, cash, holdings, trade records
        :param rebalance_date:
        :return:
        """
        today_date = rebalance_date
        yesterday_date = self.combined_total_return.index[self.combined_total_return.index.get_loc(today_date) - 1]

        # get the price on rebalance date
        price = self.price_data.loc[rebalance_date, :]
        # get the holdings on rebalance date
        holdings = self.holdings.loc[rebalance_date, :]
        holdings_yesterday = self.holdings.loc[yesterday_date, :]
        holdings = holdings_yesterday.copy()
        # get the total return on rebalance date
        total_return_1d = self.combined_total_return.loc[yesterday_date:rebalance_date, :].pct_change().iloc[-1, :]
        # get the portfolio account on rebalance date
        portfolio_account = self.portfolio_account.loc[rebalance_date, :]
        portfolio_account_yesterday = self.portfolio_account.loc[yesterday_date, :]
        portfolio_account['Unavailable_Cash'] = portfolio_account_yesterday['Unavailable_Cash'].copy()
        portfolio_account['Available_Cash'] = portfolio_account_yesterday['Available_Cash'] * (
                1 + total_return_1d.loc['CASH'])
        portfolio_account['Income'] = (portfolio_account_yesterday['Income'] +
                                       portfolio_account_yesterday['Available_Cash'] * (total_return_1d.loc['CASH']))
        # get the dividend on rebalance date
        if today_date in self.combined_dividend.index.values:
            dividend = self.combined_dividend.loc[self.combined_dividend.index.isin([rebalance_date]),
            self.combined_dividend.columns[self.combined_dividend.columns.isin(holdings.index)]]
            dividend = (dividend * holdings.loc[dividend.columns]).sum().sum()
            portfolio_account['Unavailable_Cash'] = (portfolio_account_yesterday['Unavailable_Cash'] + dividend)
        else:
            dividend = None
        if today_date in self.combined_dividend['PayDate'].values:
            payout_information = self.combined_dividend.loc[self.combined_dividend['PayDate'].isin([rebalance_date])]
            exdate_holding = self.holdings.loc[payout_information.index[0], :]
            payout_dividend = payout_information.loc[:, payout_information.columns.isin(exdate_holding.index)]
            payout_dividend = (payout_dividend * exdate_holding.loc[payout_dividend.columns]).sum().sum()
            portfolio_account['Available_Cash'] = (portfolio_account['Available_Cash'] + payout_dividend)
            portfolio_account['Unavailable_Cash'] = (portfolio_account['Unavailable_Cash'] - payout_dividend)
            portfolio_account['Income'] = (portfolio_account['Income'] + payout_dividend)
        else:
            payout_dividend = None
        holdings['CASH'] = portfolio_account['Available_Cash']
        # update nav
        portfolio_account['NAV'] = (holdings * price[holdings.index] * holdings.index.map(self.contract_unit_dict)
                                    ).sum() + portfolio_account['Unavailable_Cash']
        # update holdings
        self.holdings.loc[rebalance_date, :] = holdings
        # update portfolio account
        self.portfolio_account.loc[rebalance_date, :] = portfolio_account
        # update trade records
        pass

    def backtesting(self, portfolio_construct_method='trend_following', is_generate_output=True,
                    is_generate_performance_analysis=True):
        """
        loop through each day of trading day
        decide whether to rebalance the portfolio based on the signal
        fill history floating weight
        fill history portfolio holdings
        record trade records
        """
        if self.rebalance_freq is not None:
            potential_rebalance_date = self.rebalance_time_series(self.combined_total_return.index, self.start_date,
                                                                  self.end_date, self.rebalance_freq)
            self.__potential_rebalance_date = potential_rebalance_date.copy()
        # loop through each day of trading day
        # wrapper for loop with tqdm
        for trade_date in tqdm.tqdm(self.combined_total_return.index, desc='Backtesting Strategy'):
        # for trade_date in self.combined_total_return.index:
            # decide whether to rebalance the portfolio based on the signal
            if self.rebalance_freq is None:
                if trade_date > pd.to_datetime(self.start_date):
                    self.portfolio_prerebalance_accounting(trade_date)
                if portfolio_construct_method == 'trend_following':
                    self.trend_following_portfolio_construction(trade_date)
                else:
                    raise Warning("Portfolio construction method does not support now")
            elif self.rebalance_freq is not None:
                if trade_date > pd.to_datetime(self.start_date):
                    self.portfolio_prerebalance_accounting(trade_date)
                if trade_date in potential_rebalance_date.index:
                    if portfolio_construct_method == 'Collar':
                        self.collar_portfolio_construction(trade_date)
                    else:
                        raise Warning("Portfolio construction method does not support now")
                else:
                    pass
            else:
                raise Warning("Rebalance frequency only support None now")
        if is_generate_output:
            self.generate_trade_output()
        if is_generate_performance_analysis:
            self.generate_performance_analysis()
        print('Backtesting finished for portfolio: ', self.name)

    def generate_trade_output(self):
        """
        generate trade output
        :return:
        """
        # generate trade output
        output_path = './output/{portfolio_name}'.format(portfolio_name=self.name)
        if os.path.exists(output_path):
            pass
        else:
            os.mkdir(output_path)
        self.trade_records.to_csv(output_path + '/trade_records.csv')
        self.portfolio_account.to_csv(output_path + '/portfolio_account.csv')
        self.holdings.to_csv(output_path + '/holdings.csv')
        print('Trade output generated for portfolio: ', self.name)

    def generate_performance_analysis(self):
        """
        generate performance analysis
        it calculates the portfolio performance based on NAV and benchmark
        the output includes:
        perfromance chart of portfolio and benchmark including active return and cumulative active return
        an Excel file of performance analysis:
        whole period and by calendar year result of return, active return, volatility, sharpe ratio,
        max drawdown, max drawdown duration, information ratio, beta, upside capture, downside capture, tunover
        :return:
        """
        combined_data = self.portfolio_account[['NAV']].merge(self.benchmark, how='left', left_index=True,
                                                              right_index=True)
        combined_data = combined_data.fillna(method='ffill').dropna()
        # both start columns from initial capital
        combined_data[self.benchmark_name] = ((combined_data[self.benchmark_name].pct_change().fillna(0) + 1).cumprod()
                                              * self.initial_capital)
        combined_data['Active Total Return Index'] = ((combined_data['NAV'].pct_change()
                                                       - combined_data[self.benchmark_name].pct_change()).fillna(0) + 1
                                                      ).cumprod()
        # create an Excel to store all the analysis by tab
        output_path = './output/{portfolio_name}'.format(portfolio_name=self.name)
        if os.path.exists(output_path):
            pass
        else:
            os.mkdir(output_path)
        writer = pd.ExcelWriter(output_path + '/performance_analysis.xlsx', engine='xlsxwriter')
        # create a performance 2 panel charts and insert into the Excel tab called 'Performance Chart'
        # up panel: portfolio and benchmark cumulative return
        # down panel: portfolio and benchmark active return with the same x scale as up panel
        # create plot
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))
        # up panel
        axes[0].plot(combined_data.index, combined_data['NAV'], label='Portfolio')
        axes[0].plot(combined_data.index, combined_data[self.benchmark_name], label='Benchmark')
        axes[0].set_ylabel('Cumulative PnL')
        axes[0].legend(loc='best')
        # down panel
        axes[1].plot(combined_data.index, combined_data['NAV'] - combined_data[self.benchmark_name],
                     label='Active PnL')
        axes[1].set_ylabel('Active PnL')
        axes[1].legend(loc='best')
        # save figure
        fig.savefig(output_path + '/performance_chart.png')
        # insert figure into Excel
        worksheet = writer.book.add_worksheet('Performance Chart')
        worksheet.insert_image('A1', output_path + '/performance_chart.png')
        # create a performance analysis table and insert into the Excel tab called 'Performance Analysis'

        # calculate performance analysis
        calendar_year = combined_data.resample('Y').last().index.year
        combined_data_archive = combined_data.copy()
        for periods in ['Whole Period'] + calendar_year.tolist():
            if periods == 'Whole Period':
                combined_data = combined_data_archive.copy()
            else:
                combined_data = combined_data_archive.loc[combined_data_archive.index.year == periods, :]
            # sub period result
            whole_period_result = pd.DataFrame(index=['Period', 'Return', 'Active Return', 'Volatility', 'Sharpe Ratio',
                                                      'Max Drawdown', 'Max Drawdown Duration', 'Information Ratio',
                                                      'Beta', 'Up Capture', 'Down Capture', '2 Way Turnover'])
            whole_period_result.loc['Period', 'Portfolio'] = str(combined_data.index[0].date()) + ' - ' + str(
                combined_data.index[-1].date())
            # annualized return
            total_days = (combined_data.index[-1] - combined_data.index[0]).days
            ptf_annualized_return = data_handler.DataHandler.total_return_calc(combined_data['NAV'], is_annualize=True)
            bmk_annualized_return = data_handler.DataHandler.total_return_calc(combined_data[self.benchmark_name],
                                                                               is_annualize=True)
            whole_period_result.loc['Return', 'Portfolio'] = ptf_annualized_return
            whole_period_result.loc['Return', 'Benchmark'] = bmk_annualized_return
            # active return
            whole_period_result.loc['Active Return', 'Portfolio'] = whole_period_result.loc['Return', 'Portfolio'] - \
                                                                    whole_period_result.loc['Return', 'Benchmark']
            # volatility
            ptf_volatility = data_handler.DataHandler.volatility_calc(combined_data['NAV'])
            bmk_volatility = data_handler.DataHandler.volatility_calc(combined_data[self.benchmark_name])
            whole_period_result.loc['Volatility', 'Portfolio'] = ptf_volatility
            whole_period_result.loc['Volatility', 'Benchmark'] = bmk_volatility
            # sharpe ratio
            whole_period_result.loc['Sharpe Ratio', 'Portfolio'] = ptf_annualized_return / ptf_volatility
            whole_period_result.loc['Sharpe Ratio', 'Benchmark'] = bmk_annualized_return / bmk_volatility

            # max drawdown
            whole_period_result.loc['Max Drawdown', 'Portfolio'] = data_handler.DataHandler.max_drawdown_calc(
                combined_data['NAV'])
            whole_period_result.loc['Max Drawdown', 'Benchmark'] = data_handler.DataHandler.max_drawdown_calc(
                combined_data[self.benchmark_name])
            # active return drawdown is using active return index rather the active PnL to calculate
            whole_period_result.loc['Max Drawdown', 'Active'] = data_handler.DataHandler.max_drawdown_calc(
                combined_data['Active Total Return Index'])

            # max drawdown duration: the longest time when the portfolio is below the previous peak
            temp_data = data_handler.DataHandler.max_drawdown_duration_calc(combined_data['NAV'])
            str_drawdown_duration = (temp_data.iloc[-1].astype(str) + ' Days till '
                                     + temp_data.index[-1].strftime('%Y-%m-%d'))
            whole_period_result.loc['Max Drawdown Duration', 'Portfolio'] = str_drawdown_duration
            temp_data = data_handler.DataHandler.max_drawdown_duration_calc(combined_data[self.benchmark_name])
            str_drawdown_duration = (temp_data.iloc[-1].astype(str) + ' Days till '
                                     + temp_data.index[-1].strftime('%Y-%m-%d'))
            whole_period_result.loc['Max Drawdown Duration', 'Benchmark'] = str_drawdown_duration
            temp_data = data_handler.DataHandler.max_drawdown_duration_calc(combined_data['Active Total Return Index'])
            str_drawdown_duration = (temp_data.iloc[-1].astype(str) + ' Days till '
                                     + temp_data.index[-1].strftime('%Y-%m-%d'))
            whole_period_result.loc['Max Drawdown Duration', 'Active'] = str_drawdown_duration

            # information ratio: portfolio return - benchmark return / portfolio return volatility
            whole_period_result.loc['Information Ratio', 'Portfolio'] = data_handler.DataHandler.information_ratio_calc(
                combined_data['NAV'].pct_change(), combined_data[self.benchmark_name].pct_change())
            # beta: covariance of portfolio and benchmark / benchmark volatility
            whole_period_result.loc['Beta', 'Portfolio'] = data_handler.DataHandler.beta_calc(
                combined_data['NAV'].pct_change(), combined_data[self.benchmark_name].pct_change())
            # up capture: portfolio up market return / benchmark up market return
            up_cap, down_cap = data_handler.DataHandler.up_down_capture_calc(combined_data['NAV'].pct_change(),
                                                                             combined_data[self.benchmark_name
                                                                             ].pct_change())
            whole_period_result.loc['Up Capture', 'Portfolio'] = up_cap
            # down capture: portfolio down market return / benchmark down market return
            whole_period_result.loc['Down Capture', 'Portfolio'] = down_cap
            # 2-way turnover: portfolio holding change as percentage of total portfolio value at the time
            # initial portfolio launch is turned is excluded
            if hasattr(self, 'trade_records') & hasattr(self, 'price_data'):
                self.trade_records['Trade Date'] = pd.to_datetime(self.trade_records['Trade Date'])
                temp_data = self.trade_records[['Trade Date', 'Master_Ticker', 'Amount']].merge(
                    self.price_data.unstack().to_frame('Price').reset_index(), how='left',
                    left_on=['Master_Ticker', 'Trade Date'], right_on=['level_0', 'Date']).drop(
                    columns=['level_0', 'Date'])
                # merge contract unit
                temp_data['Contract_Unit'] = temp_data['Master_Ticker'].map(self.contract_unit_dict)
                temp_data['Trade Value'] = temp_data['Amount'] * temp_data['Price'] * temp_data['Contract_Unit']
                temp_data = temp_data.merge(self.portfolio_account[['NAV']], how='left', left_on='Trade Date',
                                            right_index=True)
                temp_data['Trade Value Pct'] = temp_data['Trade Value'] / temp_data['NAV']
                whole_period_result.loc['2 Way Turnover', 'Portfolio'] = (temp_data['Trade Value Pct'].iloc[1:].abs().sum()
                                                                          / (total_days / 365))
            elif self.holdings_by_weight is not None:
                temp_data = self.holdings_by_weight.pivot_table(index='Date', columns='Master_Ticker', values='Weight')
                whole_period_result.loc['2 Way Turnover', 'Portfolio'] = (temp_data.diff().dropna().abs().sum().sum()
                                                                          / (total_days / 365))
            else:
                raise Warning("Trade records or holdings by weight are not available")
                whole_period_result.loc['2 Way Turnover', 'Portfolio'] = None
            # insert result into Excel
            whole_period_result.to_excel(writer, sheet_name=str(periods))
        # save Excel
        writer.book.close()
        writer.save()
        print('Performance analysis generated for portfolio: ', self.name)

    @staticmethod
    def rebalance_time_series(time_series_index, start_date, end_date, rebalance_freq='D'):
        """
        rebalance time series data
        :param data: pandas dataframe
        :param rebalance_freq: str
        :param start_date: str
        :param end_date: str
        :return: pandas dataframe
        """
        time_series_index = time_series_index.to_frame().sort_index()
        time_series_index.columns = ['Rebalance Date']
        time_series_index = time_series_index.loc[start_date:end_date, :]
        if rebalance_freq is None:
            return time_series_index
        elif rebalance_freq == 'D':
            time_series_index = data_handler.DataHandler.remove_non_business_date(time_series_index)
            return time_series_index
        elif rebalance_freq == 'W':
            time_series_index = data_handler.DataHandler.remove_non_business_date(time_series_index)
            # every friday
            time_series_index = time_series_index.resample('W-FRI').last().dropna()
            return time_series_index
        elif rebalance_freq == 'M':
            time_series_index = data_handler.DataHandler.remove_non_business_date(time_series_index)
            time_series_index = time_series_index.resample('BM').last().dropna()
            return time_series_index
        elif rebalance_freq == 'Q':
            time_series_index = data_handler.DataHandler.remove_non_business_date(time_series_index)
            time_series_index = time_series_index.resample('BQ').last().dropna()
            return time_series_index
        elif rebalance_freq == 'Y':
            time_series_index = data_handler.DataHandler.remove_non_business_date(time_series_index)
            time_series_index = time_series_index.resample('Y').last().dropna()
            return time_series_index
        elif rebalance_freq == 'M3F':
            # M3F: every third Friday of the month
            time_series_index = data_handler.DataHandler.remove_non_business_date(time_series_index)
            # get the third Friday of the month
            time_series_index = time_series_index.resample('WOM-3FRI').last()
            time_series_index = time_series_index.index.to_frame().sort_index()
            time_series_index.columns = ['Rebalance Date']
            return time_series_index
        else:
            raise ValueError("Rebalance frequency not supported")
