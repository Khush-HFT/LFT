import numpy as np
import pandas as pd
from pandas import Timestamp
import xarray as xr
import fmpsdk
from datetime import datetime as dt
import datetime as d
import requests
from concurrent.futures import ThreadPoolExecutor

class DataLoader:
    # @staticmethod
    # def generate_column_names(start_time_str="09:20", end_time_str="15:30", interval_minutes=10):
    #     start_time = dt.strptime(start_time_str, "%H:%M")
    #     end_time = dt.strptime(end_time_str, "%H:%M")
    #     times = pd.date_range(start_time, end_time, freq=f'{interval_minutes}min').time
    #     return [time.strftime("%H:%M price") for time in times]


    @staticmethod
    def get_first_date_of_quarter(date):
        quarter_start_month = (date.month - 1) // 3 * 3 + 1
        return pd.Timestamp(year=date.year, month=quarter_start_month, day=1)

    @staticmethod
    def load_data():
        company_df = pd.read_csv('ind_nifty500list.csv')
        companies = company_df['Symbol'].values

        date_df = pd.read_csv('date.csv')
        dates = pd.to_datetime(date_df['date'])

        NSE_df = pd.read_csv('https://assets.upstox.com/market-quote/instruments/exchange/NSE.csv.gz')

        to_date = d.datetime.now().date() - d.timedelta(days=1)
        from_date = d.datetime(2020, 7, 1).date()

        stock_map = {}
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(DataLoader.fetch_stock_data, company, NSE_df, to_date, from_date): company for company in companies}
            for future in futures:
                company = futures[future]
                data = future.result()
                if data is not None:
                    stock_map[company] = data

        return companies, dates, stock_map

    @staticmethod
    def fetch_stock_data(company, NSE_df, to_date, from_date):
        instrument = NSE_df[NSE_df['tradingsymbol'] == company]
        if instrument.empty:
            return None

        url = f"https://api.upstox.com/v2/historical-candle/{instrument['instrument_key'].values[0]}/day/{to_date}/{from_date}"
        headers = {'Accept': 'application/json'}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = pd.DataFrame(response.json()['data']['candles'])
            if not data.empty:
                data.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'oi']
                data["datetime"] = pd.to_datetime(data["datetime"]) + pd.Timedelta(minutes=1)
                data["date"] = data["datetime"].dt.date
                data["date"] = pd.to_datetime(data["date"])
                # data["time"] = data["datetime"].dt.strftime('%H:%M') + ' price'
                return data.drop(columns=["datetime"])
        return None

class FinancialDataReader:
    def __init__(self, api_key):
        self.api_key = api_key

    # def get_key_metrics(self, symbol, period="quarter"):
    #     data = fmpsdk.key_metrics(apikey=self.api_key, symbol=symbol, period=period, limit=18)
    #     data2 = fmpsdk.income_statement(apikey=self.api_key, symbol=symbol, period=period, limit=18)

    #     df1 = pd.DataFrame(data)
    #     df2 = pd.DataFrame(data2)

    #     return pd.merge(df1, df2, on='date')
        # data = fmpsdk.key_metrics(apikey=self.api_key, symbol=symbol, period=period, limit=18)
        # return pd.DataFrame(data)

    def get_key_metrics(self, symbol, period="quarter"):
    # Retrieve data from fmpsdk
        data = fmpsdk.key_metrics(apikey=self.api_key, symbol=symbol, period=period, limit=18)
        data2 = fmpsdk.income_statement(apikey=self.api_key, symbol=symbol, period=period, limit=18)

        # Convert to DataFrames and merge on 'date'
        return pd.merge(pd.DataFrame(data), pd.DataFrame(data2), on='date')
def main():  # sourcery skip: remove-dict-keys

    column_names = ['open', 'high', 'low', 'close', 'volume'] #change
    companies, dates, stock_map = DataLoader.load_data()

    api_key = 'USpJqLtmwNP3brqw33DQGEramLU4VvSx'
    financial_data_reader = FinancialDataReader(api_key)

    fundamental_map = {}
    df = pd.DataFrame()
    for company in companies:
        data_df = financial_data_reader.get_key_metrics(company + '.NS')
        if data_df.empty:
            continue
        data_df['date'] = pd.to_datetime(data_df['date']) + pd.DateOffset(days=1)
        if df.empty:
            df = pd.DataFrame(data_df)
        else:   
            df = pd.concat([df, pd.DataFrame(data_df)], ignore_index=True)
        fundamental_map[company] = data_df
    df.to_csv('fundamentalData.csv', index=False)
    # data_dict = {
    #     "firmFundamentals": (("date", "company"), np.empty((len(dates), len(companies)), dtype=object)),
    # }

    data_dict = {name: (("date", "company"), np.zeros((len(dates), len(companies)))) for name in [
        "revenuePerShare", "netIncomePerShare", "operatingCashFlowPerShare", "freeCashFlowPerShare", "cashPerShare",
        "bookValuePerShare", "tangibleBookValuePerShare", "shareholdersEquityPerShare", "marketCap",
        "enterpriseValue", "peRatio", "pbRatio", "debtToEquity", "currentRatio", "interestCoverage", "roe",
        "freeCashFlowYield", "eps"
    ]}

    # for i in range(len(dates)):
    #     for j in range(len(companies)):
    #         data_dict["firmFundamentals"][1][i, j] = []

    for column_name in column_names:
        data_dict[column_name] = (("date", "company"), np.zeros((len(dates), len(companies))))

    xarray_3d = xr.Dataset(data_dict, coords={"date": dates, "company": companies})
    df = xarray_3d.to_dataframe()

    for company in companies:
        if company not in fundamental_map:
            continue
        for date in dates:
            new_date = DataLoader.get_first_date_of_quarter(date)
            if new_date in fundamental_map[company]['date'].values:
                index = fundamental_map[company][fundamental_map[company]['date'] == new_date].index[0]
                for metric in data_dict.keys():
                    if metric not in column_names:
                        df.loc[(date, company), metric] = fundamental_map[company].loc[index, metric]
                        # print(df.at[(date, company), column_name], fundamental_map[company].at[index, metric])

            filtered_df = stock_map.get(company, pd.DataFrame())
            if not filtered_df.empty:
                filtered_data = filtered_df[(filtered_df['date'] == date)]
                for column_name in column_names:
                    if not filtered_data.empty: 
                        df.at[(date, company), column_name] = filtered_data[column_name].values[0]
    # print(df)
    #print revenue per share of 3MINDIA on 2020-07-01
    
    ds = df.to_xarray()
    ds.to_netcdf('my_3d_dataarray.nc')
    # print(ds)
    # print(stock_map['3MINDIA'])
    print(ds.to_dataframe())

if __name__ == "__main__":
    main()