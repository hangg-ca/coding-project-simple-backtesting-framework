import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data import data_handler
from portfolio import portfolio
import tqdm

########################################################################################################################
# This code is only for demonstration purpose and not for real trading. Any changes are not suggested by the creator of
# this code. The creator of this code is not responsible for any loss caused by using this code.
# The user should use this code at their own risk.
# Creator: Hangg https://github.com/hangg-ca
########################################################################################################################


class Strategy(object):
    '''
    Strategy class: Strategy class as parent class for all strategies
    '''

    def __init__(self, **kwargs):
        self.name = kwargs['name']
        self.data = kwargs['data']

        # type check on price and dividend to be pandas dataframe
        if not isinstance(self.data, pd.DataFrame):
            raise TypeError("Data must be pandas dataframe")

    # void method
    def signal_generation(self):
        raise NotImplementedError("Should have implemented signal_generation()")

    # void method
    def training(self):
        # reserve for machine learning strategy
        pass

    # void method
    def prediction(self):
        # reserve for machine learning strategy
        pass

    @staticmethod
    def signal_test_ic(signal, forward_return):
        """
        signal_test_ic method: calculate the information coefficient of the signal with different forward return could
        help detect the signal decay issue and best time horizon for the signal
        sigal: pandas dataframe of signal with columns of date, master_ticker, signal
        forward_return: pandas dataframe of forward return with columns of date, master_ticker, forward_return of different time horizon
        """
        combined_data = signal.merge(forward_return, how='left', on=['Date', 'Master_Ticker'])
        ic_result = combined_data[['signal'] + forward_return.columns.tolist()[2:]].corr('spearman').iloc[0, 1:]
        ic_result_by_year = combined_data[['signal'] + forward_return.columns.tolist()[2:]].groupby(
            combined_data['Date'].dt.year).corr('spearman').iloc[0::2, 1:]
        return ic_result, ic_result_by_year


class MovingAverageCrossStrategy(Strategy):
    """
    MovingAverageCrossStrategy class: Moving Average Cross Strategy class as child class of Strategy class
    if 50-day moving average crosses above 200-day moving average,
    generate a buy signal;
    if 50-day moving average crosses below 200-day moving average,
    generate a sell signal;
    if nothing happened, generate a neutral signal.
    """

    def __init__(self, **kwargs):
        Strategy.__init__(self, **kwargs)
        self.short_window = kwargs['short_window']
        self.long_window = kwargs['long_window']
        self.start_date = kwargs['start_date']
        self.end_date = kwargs['end_date']
        if self.short_window > self.long_window:
            raise ValueError("Short window must be less than long window")
        if self.short_window < 1:
            raise ValueError("Short window must be greater than 1")
        if self.long_window < 1:
            raise ValueError("Long window must be greater than 1")
        if 'price_col_name' not in kwargs.keys():
            self.price_col_name = 'Adj_Price'
        else:
            self.price_col_name = kwargs['price_col_name']
        self.__strategy_type = 'Trend Following'

    def signal_generation(self):
        """
        signal_generation method: calculate the signal of moving average cross strategy
        self.data: pandas dataframe of price with columns of date, master_ticker, price
        """
        # calculate moving average
        self.data = data_handler.DataHandler.remove_non_business_date(self.data, if_index_date=True)
        self.data.sort_index(inplace=True)
        self.data['short_ma'] = self.data[self.price_col_name].rolling(window=self.short_window,
                                                                       min_periods=self.short_window).mean()
        self.data['long_ma'] = self.data[self.price_col_name].rolling(window=self.long_window,
                                                                      min_periods=self.long_window).mean()
        # calculate signal short_ma > long_ma: 1, short_ma < long_ma: -1
        self.data['signal'] = np.where(self.data['short_ma'] > self.data['long_ma'], 1, -1)
        self.signal_history = self.data[
            ['signal', 'master_ticker', 'short_ma', 'long_ma', self.price_col_name]].dropna()
        self.signal_history = self.signal_history[(self.signal_history.index >= self.start_date) &
                                                  (self.signal_history.index <= self.end_date)]
        return self.signal_history

    def signal_plot(self, master_ticker):
        """
        plot buy sell opportunity alongside the price line chart for assigned master_ticker.
        the buy and sell signal only showing the first signal of the same type before it changed to another signal.
        to reduce the image redundancy. The signal is marked above the price line chart.
        the buy signal is marked as small green triangle and the sell signal is marked as red triangle.
        the x axis is date and the y axis is price.
        """
        # select data for assigned master_ticker
        plot_data = self.signal_history[self.signal_history['master_ticker'] == master_ticker].reset_index()
        # remove the duplicate signal
        plot_data.loc[plot_data['signal'].diff() == 0, 'signal'] = np.nan
        # plot figure
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        ax.plot(plot_data['Date'], plot_data[self.price_col_name], label='Price')
        # plot buy signal traingle
        ax.scatter(plot_data['Date'][plot_data['signal'] == 1].tolist(),
                   plot_data[self.price_col_name][plot_data['signal'] == 1].tolist(),
                   marker='^', color='g', s=100, label='Buy Signal')
        # plot sell signal triangle
        ax.scatter(plot_data['Date'][plot_data['signal'] == -1].tolist(),
                   plot_data[self.price_col_name][plot_data['signal'] == -1].tolist(),
                   marker='v', color='r', s=100, label='Sell Signal')
        # # plot short moving average
        # ax.plot(plot_data['Date'], plot_data['short_ma'], label='Short Moving Average')
        # # plot long moving average
        # ax.plot(plot_data['Date'], plot_data['long_ma'], label='Long Moving Average')
        ax.legend(loc='best')
        ax.set_ylabel('Price')
        ax.set_xlabel('Date')
        ax.set_title('Moving Average Cross Strategy Signal Check')

        # save figure
        if not os.path.exists('./output'):
            os.mkdir('./output')
        fig.savefig('./output/' + master_ticker + '_Moving_Average_Cross_Strategy_Signal_Check.png')
        return


class RSI(Strategy):
    """
    RSI class: RSI Strategy class as child class of Strategy class
    if RSI > 70, generate a sell signal;
    if RSI < 30, generate a buy signal;
    if nothing happened, generate a neutral signal.
    """

    def __init__(self, **kwargs):
        Strategy.__init__(self, **kwargs)
        self.window = kwargs['window']
        self.start_date = kwargs['start_date']
        self.end_date = kwargs['end_date']
        if self.window < 1:
            raise ValueError("Window must be greater than 1")
        if 'price_col_name' not in kwargs.keys():
            self.price_col_name = 'Adj_Price'
        else:
            self.price_col_name = kwargs['price_col_name']
        self.__strategy_type = 'Mean Reversion'

    def signal_generation(self):
        """
        signal_generation method: calculate the signal of RSI strategy
        self.data: pandas dataframe of price with columns of date, master_ticker, price
        The RSI is always between 0 and 100, with stocks above 70 considered overbought and stocks below 30 oversold.
        Divergence between the price and RSI can also be analysed for potential reversals.

        Calculation

        RS = Average Gain in the Period / Average Loss in the Period

        RSI = 100 - (100 / (1 + RS))

        Average Gain is calculated as
        (Previous Average Gain * (Period - 1) + Current Gain) / Period except for the first day
        which is just an SMA. The Average Loss is similarly calculated using Losses.
        """
        # calculate RSI
        self.data = data_handler.DataHandler.remove_non_business_date(self.data, if_index_date=True)
        self.data.sort_index(inplace=True)
        delta = self.data[self.price_col_name].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=self.window).mean()
        avg_loss = loss.rolling(window=self.window).mean()
        rs = avg_gain / avg_loss
        self.data['rsi'] = 100 - (100 / (1 + rs))
        # calculate signal RSI > 70: -1, RSI < 30: 1
        self.data['signal'] = np.where(self.data['rsi'] > 70, -1, np.where(self.data['rsi'] < 30, 1, 0))
        self.signal_history = self.data[['signal', 'master_ticker', 'rsi', self.price_col_name]].dropna()
        self.signal_history = self.signal_history[(self.signal_history.index >= self.start_date) &
                                                  (self.signal_history.index <= self.end_date)]
        return self.signal_history

    def signal_plot(self, master_ticker):
        """
        plot buy sell opportunity alongside the price line chart for assigned master_ticker.
        the buy and sell signal only showing the first signal of the same type before it changed
        to another signal. to reduce the image redundancy. The signal is marked above the price line chart.
        the buy signal is marked as small green triangle and the sell signal is marked as red triangle.
        the x axis is date and the y axis is price.
        """
        # select data for assigned master_ticker
        plot_data = self.signal_history[self.signal_history['master_ticker'] == master_ticker].reset_index()
        # remove the duplicate signal
        plot_data.loc[plot_data['signal'].diff() == 0, 'signal'] = np.nan
        # plot figure
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        ax.plot(plot_data['Date'], plot_data[self.price_col_name], label='Price')
        # plot buy signal traingle
        ax.scatter(plot_data['Date'][plot_data['signal'] == 1].tolist(),
                   plot_data[self.price_col_name][plot_data['signal'] == 1].tolist(),
                   marker='^', color='g', s=100, label='Buy Signal')
        # plot sell signal triangle
        ax.scatter(plot_data['Date'][plot_data['signal'] == -1].tolist(),
                   plot_data[self.price_col_name][plot_data['signal'] == -1].tolist(),
                   marker='v', color='r', s=100, label='Sell Signal')
        # # plot RSI
        # ax.plot(plot_data['Date'], plot_data['rsi'], label='RSI')
        ax.legend(loc='best')
        ax.set_ylabel('Price')
        ax.set_xlabel('Date')
        ax.set_title('RSI Strategy Signal Check')

        # save figure
        if not os.path.exists('./output'):
            os.mkdir('./output')
        fig.savefig('./output/' + master_ticker + '_RSI_Strategy_Signal_Check.png')
        return


class Collar(Strategy):
    """
    Collar class: Collar Strategy class as child class of Strategy
    a)	Long SPY
    b)	Buy SPY 95% strike OTM put.
    The put notional is 100% of the long position to provide protection.
    The maturities of the contracts are evenly spread across 3, 6, 9 and 12-month.
    The positions are rolled every 3 months to keep the structure stable.
    c)	Sell SPY 105% strike OTM call.
    The call notional is 100% of the long position to generate income.
    The maturity of the contracts is 1 month. They are rolled every month when they expire.
    """

    def __init__(self, **kwargs):
        Strategy.__init__(self, **kwargs)
        self.start_date = kwargs['start_date']
        self.end_date = kwargs['end_date']
        self.__strategy_type = 'Hedge'
        self.underlying_ticker = kwargs['underlying_ticker']
        self.rebalance_freq = kwargs['rebalance_freq']
        self.data = data_handler.DataHandler.remove_non_business_date(self.data, if_index_date=True)
        self.data = self.data[(self.data.index >= self.start_date) & (self.data.index <= self.end_date)]
        self.put_maturity_basket = kwargs['put_maturity_basket']
        self.call_maturity_basket = kwargs['call_maturity_basket']
        self.put_moneyness = kwargs['put_moneyness']
        self.call_moneyness = kwargs['call_moneyness']

    def signal_generation(self):
        """
        signal_generation method: calculate the signal of collar strategy
        1: long, -1: short, 0: neutral
        """
        signal = pd.DataFrame(index=self.data.index)
        potential_rebalance_dates = portfolio.MultiAssetPortfolio.rebalance_time_series(self.data.index,
                                                                                        self.start_date,
                                                                                        self.end_date,
                                                                                        self.rebalance_freq)
        intend_put_purchase_date = potential_rebalance_dates[::3]['Rebalance Date']
        intend_call_write_date = potential_rebalance_dates['Rebalance Date']
        # wrapper with tqdm to show the progress bar and with title 'Collar Strategy Signal Generation'
        for trade_date in tqdm.tqdm(self.data.index, desc='Collar Strategy Signal Generation'):
        # for trade_date in self.data.index:
            # inherit the previous signal first
            if trade_date == pd.to_datetime(self.start_date):
                # first trade date, no previous signal
                signal.loc[trade_date, self.underlying_ticker] = 1
            else:
                last_day = signal.index[signal.index < trade_date][-1]
                signal.loc[trade_date, :] = signal.loc[last_day, :].copy()
            if trade_date in potential_rebalance_dates['Rebalance Date']:
                if trade_date in intend_put_purchase_date:
                    # all the PUT option overwrite to 0 first
                    signal.loc[trade_date, signal.columns.str.contains('OPTION_PUT')] = 0
                    for maturity in self.put_maturity_basket:
                        maturity_date = trade_date + pd.offsets.BMonthEnd(maturity)
                        strike_price = int(self.data.loc[trade_date, 'Price'] * self.put_moneyness/100)
                        signal.loc[
                            trade_date, 'OPTION_PUT_' + self.underlying_ticker + '_' + str(self.put_moneyness) +
                                           '_' + str(strike_price) + '_' + maturity_date.strftime('%Y%m%d')] = 1
                if trade_date in intend_call_write_date:
                    # all the CALL option overwrite to 0 first
                    signal.loc[trade_date, signal.columns.str.contains('OPTION_CALL')] = 0
                    for maturity in self.call_maturity_basket:
                        maturity_date = trade_date + pd.offsets.BMonthEnd(maturity)
                        strike_price = int(self.data.loc[trade_date, 'Price'] * self.call_moneyness /100)
                        signal.loc[
                            trade_date, 'OPTION_CALL_' + self.underlying_ticker + '_' + str(self.call_moneyness) +
                                           '_' + str(strike_price) + '_' + maturity_date.strftime('%Y%m%d')] = -1
        signal.columns.name = 'master_ticker'
        signal = signal.unstack().reset_index().rename(columns={0: 'signal'}).set_index('Date')
        signal.dropna(inplace=True)
        self.signal_history = signal
        return self.signal_history
