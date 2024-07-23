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


    def get_key_metrics(self, symbol, period="quarter"):
    # Retrieve data from fmpsdk
        data1 = fmpsdk.key_metrics(apikey=self.api_key, symbol=symbol, period=period, limit=18)
        data2 = fmpsdk.income_statement(apikey=self.api_key, symbol=symbol, period=period, limit=18)
        data3 = fmpsdk.financial_ratios(apikey=self.api_key, symbol=symbol, period=period, limit = 18)
        data4 = fmpsdk.company_profile(apikey=self.api_key, symbol=symbol)
        data1 = pd.DataFrame(data1)
        data2 = pd.DataFrame(data2)
        data3 = pd.DataFrame(data3)
        data4 = pd.DataFrame(data4)


        
        common_cols = set(data1.columns) & set(data2.columns) - {'date'}
        data2 = data2.drop(columns=common_cols)
        result_df = pd.merge(data1, data2, on='date')
        common_cols = set(result_df.columns) & set(data3.columns) - {'date'}
        data3 = data3.drop(columns=common_cols)
        result_df = pd.merge(result_df, data3, on='date')

        result_df['sector'] = data4['sector'].values[0]

        return result_df, data4
    

def process_data():  # sourcery skip: remove-dict-keys

    column_names = ['open', 'high', 'low', 'close', 'volume'] #change
    companies, dates, stock_map = DataLoader.load_data()

    api_key = 'USpJqLtmwNP3brqw33DQGEramLU4VvSx'
    financial_data_reader = FinancialDataReader(api_key)

    fundamental_map = {}
    df = pd.DataFrame()
    s_df = pd.DataFrame()
    for company in companies:
        data_df, sector_df = financial_data_reader.get_key_metrics(company + '.NS')

        if s_df.empty:
            s_df = pd.DataFrame(sector_df)
        else:   
            s_df = pd.concat([s_df, pd.DataFrame(sector_df)], ignore_index=True)
        

        if data_df.empty:
            continue

        data_df['date'] = pd.to_datetime(data_df['date']) + pd.DateOffset(days=1)
        if df.empty:
            df = pd.DataFrame(data_df)
        else:   
            df = pd.concat([df, pd.DataFrame(data_df)], ignore_index=True)
        fundamental_map[company] = data_df
    df.to_csv('fundamentalData.csv', index=False)
    s_df.to_csv('sectorData.csv', index=False)
    

    data_dict = {
    name: (("date", "company"), np.zeros((len(dates), len(companies))))
    for name in [
        "netProfitMargin", "revenuePerShare", "netIncomePerShare", "operatingCashFlowPerShare",
        "freeCashFlowPerShare", "cashPerShare", "bookValuePerShare", "tangibleBookValuePerShare",
        "shareholdersEquityPerShare", "marketCap", "enterpriseValue", "peRatio", "priceToSalesRatio",
        "pocfratio", "pfcfRatio", "pbRatio", "evToSales", "enterpriseValueOverEBITDA",
        "evToOperatingCashFlow", "earningsYield", "freeCashFlowYield", "debtToEquity", "debtToAssets",
        "netDebtToEBITDA", "interestCoverage", "incomeQuality", "dividendYield", "payoutRatio",
        "salesGeneralAndAdministrativeToRevenue", "returnOnTangibleAssets", "workingCapital",
        "tangibleAssetValue", "netCurrentAssetValue", "averageReceivables", "receivablesTurnover",
        "capexPerShare", "quickRatio", "cashRatio", "grossProfitMargin", "returnOnAssets",
        "returnOnCapitalEmployed", "companyEquityMultiplier", "netIncomePerEBT", "longTermDebtToCapitalization",
        "totalDebtToCapitalization", "fixedAssetTurnover", "operatingCashFlowPerShare", "freeCashFlowPerShare",
        "cashFlowCoverageRatios", "shortTermCoverageRatios", "capitalExpenditureCoverageRatio",
        "dividendPaidAndCapexCoverageRatio", "daysOfSalesOutstanding", "daysOfInventoryOutstanding",
        "operatingCycle", "daysOfPayablesOutstanding", "cashConversionCycle", "operatingProfitMargin",
        "pretaxProfitMargin", "netProfitMargin", "effectiveTaxRate", "ebtPerEbit", "debtRatio",
        "debtEquityRatio", "cashFlowToDebtRatio", "assetTurnover", "priceEarningsToGrowthRatio",
        "enterpriseValueMultiple", "priceFairValue"
    ]
}

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

                        if metric == 'netProfitMargin':
                            df.loc[(date, company), metric] = fundamental_map[company].loc[index, 'netIncome'] / fundamental_map[company].loc[index, 'revenue']
                        else:
                            df.loc[(date, company), metric] = fundamental_map[company].loc[index, metric]

            filtered_df = stock_map.get(company, pd.DataFrame())
            if not filtered_df.empty:
                filtered_data = filtered_df[(filtered_df['date'] == date)]
                for column_name in column_names:
                    if not filtered_data.empty: 
                        df.at[(date, company), column_name] = filtered_data[column_name].values[0]
    print(df)

    ds = df.to_xarray()


    ds.to_netcdf('my_3d_dataarray.nc')

if __name__ == "__main__":
    process_data()