import pandas as pd
import numpy as np
import pickle

########################################################################################################################
# This code is only for demonstration purpose and not for real trading. Any changes are not suggested by the creator of
# this code. The creator of this code is not responsible for any loss caused by using this code.
# The user should use this code at their own risk.
# Creator: Hangg https://github.com/hangg-ca
########################################################################################################################

class DataHandler:
    def __init__(self):
        pass

    @staticmethod
    def load_excel_data(path, sheet_name):
        """
        load excel data from path and sheet name
        :param path:
        :param sheet_name:
        :return:
        """
        raw_data = pd.read_excel(path, sheet_name=sheet_name)
        return raw_data

    @staticmethod
    def load_csv_data(path):
        raw_data = pd.read_csv(path)
        return raw_data

    @staticmethod
    def load_sql_data():
        pass

    @staticmethod
    def load_bbg_data():
        pass

    @staticmethod
    def load_pickle_data(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data

    @staticmethod
    def save_pickle_data(obj, path):
        with open(path, 'wb') as f:
            pickle.dump(obj, f)

    @staticmethod
    def load_parquet_data(path):
        # parquet is a columnar storage format, it is good for big data
        pass

    @staticmethod
    def save_parquet_data():
        pass

    @staticmethod
    def clean_time_series(raw_data, date_col, how_drop_na='any'):
        raw_data = raw_data.dropna(how=how_drop_na)
        raw_data = raw_data.reset_index(drop=True)
        raw_data[date_col] = pd.to_datetime(raw_data[date_col])
        raw_data = raw_data.set_index(date_col)
        return raw_data

    @staticmethod
    # summarize data with outlier detection function
    def data_overview(raw_data):
        data_overview = pd.DataFrame({'Data Type': raw_data.dtypes,
                                      'Missing Values': raw_data.isnull().sum(),
                                      'Unique Values': raw_data.nunique()})
        return data_overview

    @staticmethod
    def print_missing_data_pct_by_col(raw_data):
        print(raw_data.isnull().sum() / raw_data.shape[0])

    @staticmethod
    def remove_non_business_date(raw_data, if_index_date=True, date_col='Date'):
        """
        remove non-business date without dealing more complicated holiday issue
        :param raw_data:
        :param if_index_date:
        :param date_col:
        :return:
        """
        if if_index_date:
            raw_data.sort_index(inplace=True)
            raw_data = raw_data[raw_data.index.dayofweek < 5]
        else:
            raw_data.sort_values(by=date_col, inplace=True)
            raw_data = raw_data[raw_data[date_col].dt.dayofweek < 5]
        return raw_data

    @staticmethod
    def normalize_data(raw_data):
        raw_data = raw_data.dropna()
        raw_data = raw_data.reset_index(drop=True)
        raw_data = (raw_data - raw_data.mean()) / raw_data.std()
        return raw_data

    @staticmethod
    def winsorize_data(raw_data, threshold):
        raw_data = raw_data.dropna()
        raw_data = raw_data.reset_index(drop=True)
        raw_data = raw_data.clip(lower=raw_data.quantile(threshold), upper=raw_data.quantile(1 - threshold), axis=1)
        return raw_data

    @staticmethod
    def total_return_calc(input_data, is_annualize=True):
        """
        annuliaze return
        :param returns:
        :param period:
        :return:
        """
        if not is_annualize:
            return input_data.iloc[-1] / input_data.iloc[0] - 1
        else:
            total_days = (input_data.index[-1] - input_data.index[0]).days
            annualized_return = (input_data.iloc[-1] / input_data.iloc[0]) ** (365 / total_days) - 1
            return annualized_return

    @staticmethod
    def volatility_calc(input_data):
        """
        calculate volatility
        :param returns:
        :param period:
        :return:
        """
        return input_data.pct_change().std() * np.sqrt(252)

    @staticmethod
    def sharpe_ratio_calc(input_data, risk_free_rate=0.0):
        """
        calculate sharpe ratio
        :param returns:
        :param period:
        :param risk_free_rate:
        :return:
        """
        total_return = DataHandler.total_return_calc(input_data)
        total_risk = DataHandler.volatility_calc(input_data)
        return (total_return - risk_free_rate) / total_risk

    @staticmethod
    def max_drawdown_calc(input_data):
        """
        calculate max drawdown
        :param returns:
        :param period:
        :return:
        """
        return (input_data / input_data.cummax() - 1).min()

    @staticmethod
    def max_drawdown_duration_calc(input_data):
        """
        calculate max drawdown duration
        :param returns:
        :param period:
        :return:
        """
        cum_max = input_data.cummax()

        # Identify periods where NAV is below the peak
        drawdown = (input_data < cum_max)

        # Calculate the duration of each drawdown period
        drawdown_duration = drawdown.cumsum() - drawdown.cumsum().where(~drawdown).ffill().fillna(0)

        # Find the maximum duration of drawdown
        max_duration = drawdown_duration.sort_values(ascending=False).iloc[[0]]
        return max_duration

    @staticmethod
    def information_ratio_calc(input_data, benchmark):
        """
        calculate information ratio
        :param returns:
        :param benchmark:
        :param period:
        :return:
        """
        return (input_data - benchmark).mean() / (input_data - benchmark).std()

    @staticmethod
    def beta_calc(input_data, benchmark):
        """
        calculate beta
        :param returns:
        :param benchmark:
        :param period:
        :return:
        """
        return (input_data.cov(benchmark) / benchmark.var())

    @staticmethod
    def up_down_capture_calc(input_data, benchmark):
        """
        calculate up capture
        :param returns:
        :param benchmark:
        :param period:
        :return:
        """
        up_capture = input_data[benchmark>0].mean() / benchmark[benchmark>0].mean()
        down_capture = input_data[benchmark<0].mean() / benchmark[benchmark<0].mean()
        return up_capture, down_capture

    @staticmethod
    def date_convert(from_date, target_freq='D', convention='ACT/360'):
        """
        convert str date to target frequency date, liek from 1M to 30D float number
        :param from_date:
        :param target_freq:
        :return:
        """
        assert isinstance(from_date, str), 'from_date should be string!'
        if convention == 'ACT/360':
            if 'D' in target_freq:
                if 'M' in from_date:
                    converter_result = float(from_date.split('M')[0])*30
                elif 'Y' in from_date:
                    converter_result = float(from_date.split('Y')[0])*360
                else:
                    converter_result = float(from_date.split('D')[0])
            elif 'M' in target_freq:
                if 'D' in from_date:
                    converter_result = float(from_date.split('D')[0])/30
                elif 'Y' in from_date:
                    converter_result = float(from_date.split('Y')[0])*12
                else:
                    converter_result = float(from_date.split('M')[0])
            elif 'Y' in target_freq:
                if 'D' in from_date:
                    converter_result = float(from_date.split('D')[0])/360
                elif 'M' in from_date:
                    converter_result = float(from_date.split('M')[0])/12
                else:
                    converter_result = float(from_date.split('Y')[0])
        else:
            print('Convention not supported!')
        return float(converter_result)

