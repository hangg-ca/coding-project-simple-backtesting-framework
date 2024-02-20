import numpy as np
import pandas as pd
from data import data_handler
from math import log, sqrt, exp
from scipy import stats
from scipy.interpolate import LinearNDInterpolator, interp1d


class Asset(object):
    """ Asset class: Void Asset Calss as parent class for all assets like stocks, bonds, etc."""
    ' the init method should have master ticker for all the other assets'

    def __init__(self, **kwargs):
        self.name = kwargs['name']
        if not 'master_ticker' in kwargs:
            raise ValueError("Master ticker must be provided")
        self.master_ticker = kwargs['master_ticker']
        if not 'transaction_cost' in kwargs:
            self.transaction_cost = 0.0  # if not provided, set to 0
        else:
            self.transaction_cost = kwargs['transaction_cost']

    # void method
    def get_price(self):
        pass

    # void method
    def total_return_calc(self):
        pass

    def __str__(self):
        return "Asset: " + self.name + " Price: " + str(self.price) + " Amount: " + str(self.amount)


class ETF(Asset):
    """
    ETF class: ETF Asset class as child class of Asset class
    """

    def __init__(self, **kwargs):
        Asset.__init__(self, **kwargs)
        self.price = kwargs['price']
        self.dividend = kwargs['dividend']

        # type check on price and dividend to be pandas dataframe
        if not isinstance(self.price, pd.DataFrame):
            raise TypeError("Price must be pandas dataframe")
        if not isinstance(self.dividend, pd.DataFrame):
            raise TypeError("Dividend must be pandas dataframe")

    def total_return_calc(self, if_need_adj=True):
        '''
        total_return_calc method: calculate total return of ETF that includes adjusted price and dividend
        The dividend is assumed to be reinvested in the ETF at the ex-dividend date
        Todo: add stock split adjustment
        '''
        if if_need_adj:  # if adjusted price is needed
            combined_return = self.price.merge(self.dividend, how='left', left_index=True, right_index=True)
            combined_return = combined_return.fillna(0)
            combined_return['cum_div'] = combined_return['Dividends'].cumsum()
            combined_return['Adj_Price'] = combined_return['Price'] + combined_return['cum_div']
            combined_return['total_return_1d'] = combined_return['Adj_Price'].pct_change().fillna(0)
            combined_return['total_return_index'] = (combined_return['total_return_1d'] + 1).cumprod()
            # remove non-business date without dealing more complicated holiday issue
            combined_return = data_handler.DataHandler.remove_non_business_date(combined_return, if_index_date=True)
            self.total_return_index = combined_return['total_return_index']
            self.total_return_1d = combined_return['total_return_1d']
            self.__adj_price = combined_return['Adj_Price']
        else:  # if adjusted price is not needed
            self.total_return_1d = self.price.pct_change().fillna(0).iloc[:, 0].rename('total_return_1d')
            self.total_return_index = data_handler.DataHandler.remove_non_business_date(
                (self.total_return_1d + 1).cumprod().rename('total_return_index'), if_index_date=True)
            self.__adj_price = self.price
        return self.total_return_index

    def transaction_cost_calc(self, shares):
        """
        transaction_cost_calc method: calculate transaction cost of ETF

        """
        self.transaction_cost = shares * self.transaction_cost
        return self.transaction_cost

    def forward_return_generate(self, forward_return_horizon=[1, 5, 10, 20, 60, 120, 252]):
        """
        forward_return_generate method: generate forward return of ETF
        forward return could be used to test information coefficient of signal and decay of signal
        """
        # if no total return index is calculated, calculate it
        if not hasattr(self, 'total_return_index'):
            self.total_return_calc()
        # calculate total return of different horizona and then shift to get forward return
        self.forward_return = pd.DataFrame()
        for horizon in forward_return_horizon:
            self.forward_return[str(horizon) + 'd_forward_return'] = self.total_return_index.pct_change(horizon).shift(
                -horizon)
        return self.forward_return


class Cash(Asset):
    """ Cash class: Cash Asset class as child class of Asset class
    The cash rate is assumed to be the same as the market rate with ticker US00O/N Index
    """

    def __init__(self, **kwargs):
        Asset.__init__(self, **kwargs)
        self.cash_rate = kwargs['cash_rate']
        # type check on cash rate to be pandas dataframe
        if not isinstance(self.cash_rate, pd.DataFrame):
            raise TypeError("Cash rate must be pandas dataframe")
        self.cash_rate = data_handler.DataHandler.remove_non_business_date(self.cash_rate, if_index_date=True)
        self.price = pd.DataFrame({'Date': self.cash_rate.index, 'Price': 1})
        self.price = self.price.set_index('Date')
        self.currency = kwargs['currency']

    def total_return_calc(self, convention='ACT/365'):
        '''
        total_return_calc method: calculate total return of cash that includes interest rate
        '''
        if convention == 'ACT/365':
            self.total_return_1d = self.cash_rate / 100 / 365
        elif convention == 'ACT/360':
            self.total_return_1d = self.cash_rate / 100 / 360
        elif convention == 'ACT/252':
            self.total_return_1d = self.cash_rate / 100 / 252
        self.total_return_1d = self.total_return_1d.iloc[:, 0].rename('total_return_1d')
        self.total_return_index = (self.total_return_1d + 1).cumprod().rename('total_return_index')
        return self.total_return_index


class Option(Asset):
    """ Option class: Option Asset class as child class of Asset class
    """

    def __init__(self, **kwargs):
        Asset.__init__(self, **kwargs)
        self.underlying_price = kwargs['underlying_price']
        self.underlying_price = self.underlying_price.rename(
            columns={self.underlying_price.columns[0]: 'Underlying Price'})
        self.underlying_dividend = kwargs['underlying_dividend']
        self.risk_free_rate = kwargs['risk_free_rate']
        self.contract_unit = kwargs['contract_unit']
        self.transaction_cost = kwargs['transaction_cost']
        self.exercise_type = kwargs['exercise_type']
        (self.option_type, self.underlying, self.moneyness, self.strike, self.maturity_date) = self.master_ticker_parse(
            kwargs['master_ticker'])
        if 'implied_vol' in kwargs:
            self.volatility = kwargs['implied_vol']
        else:
            self.volatility = None
        if 'price' in kwargs:
            self.price = kwargs['price']
        if 'strike' in kwargs:
            self.strike = kwargs['strike']
        if 'intend_purchase_date' in kwargs:
            # this is intended trade date, Write or Buy option on this date
            self.intend_purchase_date = kwargs['intend_purchase_date']
        else:
            self.intend_purchase_date = max(pd.to_datetime(self.intend_purchase_date),
                                            self.underlying_price.index.min(), self.risk_free_rate.index.min(),
                                            self.underlying_dividend.index.min(), self.volatility.index.min())
            self.intend_purchase_date = self.intend_purchase_date.strftime('%Y-%m-%d')
        # reduce data length to the same length intend_purchase_date to maturity_date
        self.underlying_price = self.underlying_price.loc[self.intend_purchase_date:self.maturity_date]
        self.underlying_dividend = self.underlying_dividend.loc[self.intend_purchase_date:self.maturity_date]
        self.risk_free_rate = self.risk_free_rate.loc[self.intend_purchase_date:self.maturity_date]
        if self.volatility is not None:
            self.volatility = self.volatility.loc[self.intend_purchase_date:self.maturity_date]
        self.moneyness_history = (self.strike / self.underlying_price) * 100
        self.moneyness_history = self.moneyness_history.rename(
            columns={self.underlying_price.columns[0]: 'Moneyness Hist'})
        self.time_to_maturity = pd.DataFrame({'Date': self.underlying_price.index,
                                              'time_to_maturity': (pd.to_datetime(self.maturity_date)
                                                                   - self.underlying_price.index).days / 365})
        self.time_to_maturity = self.time_to_maturity.set_index('Date')

    def get_pricing(self, model='BSM', div_model='12m_continuous'):
        if model == 'BSM':
            # this code is using apply function to calculate price, which is not efficient
            # but it is easy to understand and can be improved by parallel computing apply function
            if div_model == '12m_continuous':
                div_data = self.underlying_dividend.copy()
            elif div_model == 'forward':
                raise ValueError("Forward dividend yield model is not implemented yet")
            # combined all the inputs time series into one combined dataframe
            combined_input = self.underlying_price.merge(self.time_to_maturity, how='left', left_index=True,
                                                         right_index=True)
            combined_input = combined_input.merge(div_data, how='left', left_index=True,
                                                  right_index=True)
            combined_input = combined_input.merge(self.moneyness_history, how='left', left_index=True,
                                                  right_index=True)
            combined_input.dropna(inplace=True)

            combined_input = data_handler.DataHandler.remove_non_business_date(combined_input, if_index_date=True)
            # calculate price
            combined_input['r'] = combined_input.apply(
                lambda x: Curve.get_target_tenor_rate(self.risk_free_rate,
                                                      target_tenor=float(int(x['time_to_maturity'] * 365)),
                                                      date_loc=x.name) / 100, axis=1)
            combined_input['sigma'] = combined_input.apply(
                lambda x: self.get_implied_vol(self.volatility,
                                               moneyess=x['Moneyness Hist'],
                                               maturity=float(int(x['time_to_maturity'] * 365)),
                                               date=x.name.strftime('%Y-%m-%d')) / 100, axis=1)
            combined_input['Price'] = combined_input.apply(
                lambda x: self.bsm_model(x['Underlying Price'], self.strike,
                                         x['time_to_maturity'], x['r'],
                                         x['sigma'], self.option_type,
                                         x[div_data.columns[0]] / 100), axis=1)
            self.price = combined_input[['Price']]
        elif model == 'BinomialTree':
            if div_model == '12m_continuous':
                div_data = self.underlying_dividend.copy()
            elif div_model == 'forward':
                raise ValueError("Forward dividend yield model is not implemented yet")
            combined_input = self.underlying_price.merge(self.time_to_maturity, how='left', left_index=True,
                                                         right_index=True)
            combined_input = combined_input.merge(div_data, how='left', left_index=True,
                                                  right_index=True)
            combined_input = combined_input.merge(self.moneyness_history, how='left', left_index=True,
                                                  right_index=True)
            combined_input.dropna(inplace=True)

            combined_input = data_handler.DataHandler.remove_non_business_date(combined_input, if_index_date=True)
            # calculate price
            combined_input['r'] = combined_input.apply(
                lambda x: Curve.get_target_tenor_rate(self.risk_free_rate,
                                                      target_tenor=float(int(x['time_to_maturity'] * 365)),
                                                      date_loc=x.name) / 100, axis=1)
            combined_input['sigma'] = combined_input.apply(
                lambda x: self.get_implied_vol(self.volatility,
                                               moneyess=x['Moneyness Hist'],
                                               maturity=float(int(x['time_to_maturity'] * 365)),
                                               date=x.name.strftime('%Y-%m-%d')) / 100, axis=1)
            # binomial_tree_model(S0, K, T, r, sigma, q, N, option_type='call', option_style='American')
            combined_input['Price'] = combined_input.apply(
                lambda x: self.binomial_tree_model(x['Underlying Price'], self.strike,
                                                   x['time_to_maturity'], x['r'],
                                                   x['sigma'], x[div_data.columns[0]] / 100,
                                                   100, self.option_type, self.exercise_type), axis=1)
            self.price = combined_input[['Price']]
        else:
            raise ValueError("Model not implemented yet")
        return self.price

    def total_return_calc(self):
        """
        total_return_calc method: calculate total return of option base on price
        """
        self.total_return_1d = self.price.pct_change().fillna(0).iloc[:, 0].rename('total_return_1d')
        self.total_return_index = (self.total_return_1d + 1).cumprod().rename('total_return_index')
        return self.total_return_index

    @staticmethod
    def bsm_model(S, K, T, r, sigma, option_type, dividend):
        """
        BSM method: calculate option price using Black-Scholes-Merton model
        :param S:  spot price of the underlying asset
        :param K:  strike price of the option
        :param T:  time to maturity
        :param r:  risk-free rate
        :param sigma:  volatility of the underlying asset
        :param option_type:  call or put
        :param dividend:  dividend yield
        :return:
        ref: https://www.columbia.edu/~mh2078/FoundationsFE/BlackScholes.pdf
        Warning: need statisfy assumption of BSM model and continuous dividend yield
        """
        if T == 0:
            if option_type.lower() == 'call':
                price = max(S - K, 0)
            elif option_type.lower() == 'put':
                price = max(K - S, 0)
            else:
                raise ValueError("Option type must be call or put")
        else:
            d1 = (log(S / K) + (r - dividend + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
            d2 = d1 - sigma * sqrt(T)
            if option_type.lower() == 'call':
                price = S * exp(-dividend * T) * stats.norm.cdf(d1) - K * exp(-r * T) * stats.norm.cdf(d2)
            elif option_type.lower() == 'put':
                price = K * exp(-r * T) * stats.norm.cdf(-d2) - S * exp(-dividend * T) * stats.norm.cdf(-d1)
            else:
                raise ValueError("Option type must be call or put")
        return price

    @staticmethod
    def binomial_tree_model(S0, K, T, r, sigma, q, N, option_type='call', option_style='american'):
        """
        This version is intuiative and easy to understand, but not efficient, could implement with vectorization
        S0: initial stock price
        K: strike price
        T: time to maturity
        r: risk-free interest rate
        sigma: volatility
        q: continuous dividend yield
        N: number of steps in the binomial tree
        option_type: 'call' or 'put'
        """
        # Calculate parameters
        if T == 0:
            if option_type.lower() == 'call':
                return max(S0 - K, 0)
            elif option_type.lower() == 'put':
                return max(K - S0, 0)
        else:
            dt = T / N
            u = exp(sigma * sqrt(dt))
            d = 1 / u
            p = (exp((r - q) * dt) - d) / (u - d)  # Adjusted for dividend yield

            # Initialize asset prices at maturity
            S = np.zeros((N + 1, N + 1))
            for i in range(N + 1):
                for j in range(i + 1):
                    S[j, i] = S0 * (u ** (i - j)) * (d ** j)

            # Initialize option values at maturity
            V = np.zeros((N + 1, N + 1))
            if option_type.lower() == 'call':
                V[:, N] = np.maximum(S[:, N] - K, 0)
            else:  # put
                V[:, N] = np.maximum(K - S[:, N], 0)

            # Backward induction for early exercise
            for i in range(N - 1, -1, -1):
                for j in range(i + 1):
                    # Expected value of holding
                    hold_value = exp(-r * dt) * (p * V[j, i + 1] + (1 - p) * V[j + 1, i + 1])
                    if option_type.lower() == 'call':
                        exercise_value = max(S[j, i] - K, 0)  # Exercise value
                    else:  # put
                        exercise_value = max(K - S[j, i], 0)  # Exercise value
                    if option_style.lower() == 'american':
                        V[j, i] = max(hold_value, exercise_value)
                    else:  # European
                        V[j, i] = hold_value
        return V[0, 0]

    @staticmethod
    def monte_carlo_model(**kwargs):
        """
        Monte_Carlo_model method: calculate option price using Monte Carlo model
        This is more flexible and can be used for path-dependent options, but it is also more computationally intensive
        The risk free rate and dividend yield could be model as variable series
        one reference: https://github.com/hsjharvey/Option-Pricing/blob/master/monte_carlo/monte_carlo_class.py
        """
        pass

    @staticmethod
    def other_numerical_pde_model(**kwargs):
        """
        Numerical_PDE_model method: calculate option price using Numerical PDE model
        one ref https://github.com/cantaro86/Financial-Models-Numerical-Methods/blob/master/
        2.1%20Black-Scholes%20PDE%20and%20sparse%20matrices.ipynb
        """
        pass

    def get_greeks(self, model='BSM'):
        pass

    @staticmethod
    def get_implied_vol(iv_data, moneyess=92.0, maturity=21, date='2020-01-05', method='linear'):
        """
        interpolate and extrapolate vol surface from iv_data
        return sepecific implied vol of moneyess and matruity and date
        iv data is a dataframe with index as date and columns as Value  Maturity  Moneyness
                                      Master_Ticker    Value Maturity  Moneyness
        Date
        2005-01-10   30DAY_IMPVOL_80%MNY_DF  17.0723      30D       80.0
        2005-01-11   30DAY_IMPVOL_80%MNY_DF  17.0199      30D       80.0
        """
        # create vol surce from iv_data with 3-dimension (matruity, moneyness, date)
        iv_data_local = iv_data.loc[date]
        iv_data_local = iv_data_local.pivot(index='Maturity', columns='Moneyness', values='Value')
        iv_data_local = iv_data_local.sort_index(ascending=False)

        if method == 'linear':
            # create 2-dimension grid for (matruity, moneyness)
            x = iv_data_local.columns  # moneyness
            y = iv_data_local.index  # maturity
            X, Y = np.meshgrid(x, y)
            Z = iv_data_local.values
            # create linear interpolation model
            interp_model = LinearNDInterpolator((X.flatten(), Y.flatten()), Z.flatten())
            # get the implied vol
            implied_vol = interp_model(moneyess, maturity)
            # Check if the result is NaN (meaning the point was outside the grid)
            if np.isnan(implied_vol):
                # Extrapolation is needed
                # use nearest point to get implied vol
                implied_vol = iv_data_local.iloc[np.abs(iv_data_local.index - maturity).argsort()[:1],
                np.abs(iv_data_local.columns - moneyess).argsort()[:1]].values[0][0]
            implied_vol = float(implied_vol)
        else:
            raise ValueError("Interpolation not implemented yet")
        return implied_vol

    @staticmethod
    def master_ticker_parse(master_ticker):
        """
        master_ticker_parse method: parse master ticker to get option type, underlying, moneyness, and maturity date
        # master_ticker = "OPTION" +OPTION_TYPE+underlying+moneyness+maturity_date
        # example like OPTION_CALL_SPY_120_20230120
        :param master_ticker:
        :return:
        """
        temp_component = master_ticker.split('_')
        option_type = temp_component[1]
        underlying = temp_component[2]
        moneyness = float(temp_component[3])
        strike_price = float(temp_component[4])
        maturity_date = pd.to_datetime(temp_component[5]).strftime('%Y-%m-%d')
        return option_type, underlying, moneyness, strike_price, maturity_date

    @staticmethod
    def fwd_divdend_yield_calc(input_data):
        """
        fwd_divdend_yield_calc method: calculate forward dividend yield
        :param dividend:
        :param price:
        :return:
        """
        pass

    @staticmethod
    def plot_option_vol_surface(iv_data, date_loc):
        """
        plot_option_vol_surface method: plot 3D implied vol surface
        :param iv_data:
        :param date:
        :return:
        """
        import matplotlib.pyplot as plt
        vol_data = iv_data.loc[date_loc]
        vol_data = vol_data.pivot(index='Maturity', columns='Moneyness', values='Value')
        vol_data = vol_data.sort_index(ascending=False)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x = vol_data.columns
        y = vol_data.index
        X, Y = np.meshgrid(x, y)
        Z = vol_data.values
        ax.plot_surface(X, Y, Z, cmap='viridis')
        ax.set_xlabel('Moneyness')
        ax.set_ylabel('Maturity')
        ax.set_zlabel('Implied Volatility')
        plt.show()

    @staticmethod
    def basket_option_create(time, weights):
        """
        basket_option_create method: create basket option
        :param option_list:
        :param weights:
        :return:
        """
        pass


class Curve(Asset):
    """ Curve class: Curve Asset class as child class of Asset class
    """

    def __init__(self, **kwargs):
        Asset.__init__(self, **kwargs)
        self.curve_rate = kwargs['curve_rate']
        # type check on curve rate to be pandas dataframe
        if not isinstance(self.curve_rate, pd.DataFrame):
            raise TypeError("Curve rate must be pandas dataframe")
        self.curve_rate = data_handler.DataHandler.remove_non_business_date(self.curve_rate, if_index_date=True)
        self.tenor = self.curve_rate.columns
        self.is_interpolated = False

    def interpolation(self, method='linear', taget_tenor='All', tenor_list=[5, 7, 8]):
        """
        interpolation method: interpolate the curve rate to get the rate at any tenor
        the new tenor list could be provided if empty, the dayily tenor from min to max will be used
        :param method:
        :return:
        """
        curve_rate = self.curve_rate.copy()
        curve_rate = curve_rate.reset_index()
        curve_rate = curve_rate.rename(columns={'index': 'Date'})
        curve_rate = curve_rate.melt(id_vars='Date', var_name='Tenor', value_name='Rate')
        curve_rate['Tenor'] = curve_rate['Tenor'].astype(str).str.extract('(\d+)')
        curve_rate['Tenor'] = curve_rate['Tenor'].astype(float)
        curve_rate = curve_rate.dropna()
        curve_rate = curve_rate.sort_values(by=['Date', 'Tenor'])
        curve_rate = curve_rate.drop_duplicates(subset=['Date', 'Tenor'], keep='first')
        if (taget_tenor == 'All') | (tenor_list == []):
            # generate new columns from min to max unique
            new_columns = range(curve_rate['Tenor'].astype(int).min(), curve_rate['Tenor'].astype(int).max() + 1)
            new_columns = pd.Series(new_columns).sort_values().unique()
            new_columns = new_columns.astype(float)
        else:
            new_columns = pd.Series(curve_rate['Tenor'].unique().tolist() + tenor_list).sort_values().unique()
            new_columns = new_columns.astype(float)
        curve_rate = curve_rate.pivot(index='Date', columns='Tenor', values='Rate')
        curve_rate = curve_rate.reindex(columns=new_columns)
        curve_rate = curve_rate.interpolate(method=method, axis=1)
        self.curve_rate = curve_rate
        self.is_interpolated = True
        return self.curve_rate

    @staticmethod
    def get_target_tenor_rate(curve_rate, target_tenor, date_loc, method='linear'):
        """
        Find the nearest tenor rate if the tenor is not in the curve rate
        :param tenor:
        :return:
        """
        try:
            rate_reading = curve_rate.loc[date_loc, target_tenor]
        except:
            curve_rate = curve_rate.loc[date_loc]
            curve_rate = curve_rate.dropna()

            if method == 'linear':
                interp1d_model = interp1d(curve_rate.index, curve_rate.values, kind='linear', fill_value='extrapolate')
                rate_reading = interp1d_model(target_tenor)
            else:
                raise ValueError("Interpolation not implemented yet")
            # float one number
            rate_reading = float(rate_reading)
        return rate_reading
