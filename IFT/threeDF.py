# %%
import numpy as np
import pandas as pd
from pandas import Timestamp
import xarray as xr
import fmpsdk
from datetime import datetime as dt
import datetime as d
from concurrent.futures import ThreadPoolExecutor

# %%
class DataLoader:
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

        return companies, dates

# %%
class FinancialDataReader:
    def __init__(self, api_key):
        self.api_key = api_key

    def get_key_metrics(self, symbol, period="quarter"):
        # Retrieve data from fmpsdk
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(fmpsdk.key_metrics, apikey=self.api_key, symbol=symbol, period=period, limit=43),
                executor.submit(fmpsdk.income_statement, apikey=self.api_key, symbol=symbol, period=period, limit=43),
                executor.submit(fmpsdk.financial_ratios, apikey=self.api_key, symbol=symbol, period=period, limit=43),
                executor.submit(fmpsdk.company_profile, apikey=self.api_key, symbol=symbol),
                executor.submit(fmpsdk.historical_price_full, apikey=self.api_key, symbol=symbol, from_date="2014-01-01", to_date="2024-06-28")
            ]
            data1, data2, data3, data4, data5 = [f.result() for f in futures]

        data1 = pd.DataFrame(data1)
        data2 = pd.DataFrame(data2)
        data3 = pd.DataFrame(data3)
        data4 = pd.DataFrame(data4)
        data5 = pd.DataFrame(data5)

        # Merge data1, data2, and data3 on 'date'
        common_cols = set(data1.columns) & set(data2.columns) - {'date'}
        data2.drop(columns=common_cols, inplace=True)
        result_df = pd.merge(data1, data2, on='date')

        common_cols = set(result_df.columns) & set(data3.columns) - {'date'}
        data3.drop(columns=common_cols, inplace=True)
        result_df = pd.merge(result_df, data3, on='date')

        # Add sector information
        result_df['sector'] = data4.get('sector', None)

        return result_df, data4, data5


# %%
def process_data():  

    column_names = ['open', 'high', 'low', 'close', 'volume']
    companies, dates = DataLoader.load_data()

    api_key = 'USpJqLtmwNP3brqw33DQGEramLU4VvSx'
    financial_data_reader = FinancialDataReader(api_key)


    fundamental_map = {}
    historical_map = {}
    df = pd.DataFrame()
    s_df = pd.DataFrame()
    h_df = pd.DataFrame()
    for company in companies:
        (
            data_df,
            sector_df,
            historical_df,
        ) = financial_data_reader.get_key_metrics(f'{company}.NS')

        sector_df = pd.DataFrame(sector_df)
        if s_df.empty:
            s_df = sector_df
        else:   
            s_df = pd.concat([s_df, sector_df], ignore_index=True)

        historical_df['date'] = pd.to_datetime(historical_df['date'])
        h_df = pd.DataFrame(historical_df)
        historical_map[company] = h_df

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
            "netProfitMargin", 
            "revenuePerShare",
            "netIncomePerShare",
            "operatingCashFlowPerShare",
            "freeCashFlowPerShare",
            "cashPerShare",
            "bookValuePerShare",
            "tangibleBookValuePerShare",
            "shareholdersEquityPerShare",
            "interestDebtPerShare",
            "marketCap",
            "enterpriseValue",
            "peRatio",
            "priceToSalesRatio",
            "pocfratio",
            "pfcfRatio",
            "pbRatio",
            "ptbRatio",
            "evToSales",
            "enterpriseValueOverEBITDA",
            "evToOperatingCashFlow",
            "evToFreeCashFlow",
            "earningsYield",
            "freeCashFlowYield",
            "debtToEquity",
            "debtToAssets",
            "netDebtToEBITDA",
            "currentRatio",
            "interestCoverage",
            "incomeQuality",
            "dividendYield",
            "payoutRatio",
            "salesGeneralAndAdministrativeToRevenue",
            "researchAndDdevelopementToRevenue",
            "intangiblesToTotalAssets",
            "capexToOperatingCashFlow",
            "capexToRevenue",
            "capexToDepreciation",
            "stockBasedCompensationToRevenue",
            "grahamNumber",
            "roic",
            "returnOnTangibleAssets",
            "grahamNetNet",
            "workingCapital",
            "tangibleAssetValue",
            "netCurrentAssetValue",
            "investedCapital",
            "averageReceivables",
            "averagePayables",
            "averageInventory",
            "daysSalesOutstanding",
            "daysPayablesOutstanding",
            "daysOfInventoryOnHand",
            "receivablesTurnover",
            "payablesTurnover",
            "inventoryTurnover",
            "roe",
            "capexPerShare",
            "quickRatio",
            "cashRatio",
            "daysOfSalesOutstanding",
            "daysOfInventoryOutstanding",
            "operatingCycle",
            "cashConversionCycle",
            "grossProfitMargin",
            "operatingProfitMargin",
            "pretaxProfitMargin",
            "netProfitMargin",
            "effectiveTaxRate",
            "returnOnAssets",
            "returnOnEquity",
            "returnOnCapitalEmployed",
            "netIncomePerEBT",
            "ebtPerEbit",
            "ebitPerRevenue",
            "debtRatio",
            "debtEquityRatio",
            "longTermDebtToCapitalization",
            "totalDebtToCapitalization",
            "cashFlowToDebtRatio",
            "companyEquityMultiplier",
            "fixedAssetTurnover",
            "assetTurnover",
            "operatingCashFlowSalesRatio",
            "freeCashFlowOperatingCashFlowRatio",
            "cashFlowCoverageRatios",
            "shortTermCoverageRatios",
            "capitalExpenditureCoverageRatio",
            "dividendPaidAndCapexCoverageRatio",
            "dividendPayoutRatio",
            "priceBookValueRatio",
            "priceEarningsRatio",
            "priceToFreeCashFlowsRatio",
            "priceToOperatingCashFlowsRatio",
            "priceCashFlowRatio",
            "priceEarningsToGrowthRatio",
            "enterpriseValueMultiple",
            "priceFairValue",
            "eps"
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
                for metric in data_dict:
                    if metric not in column_names:

                        if metric == 'netProfitMargin':
                            df.loc[(date, company), metric] = fundamental_map[company].loc[index, 'netIncome'] / fundamental_map[company].loc[index, 'revenue']
                        else:
                            df.loc[(date, company), metric] = fundamental_map[company].loc[index, metric]
            if date in historical_map[company]['date'].values:
                filtered_row = historical_map[company][historical_map[company]['date'] == date]

                for metric in column_names:
                    df.loc[(date, company), metric] = filtered_row[metric].values[0]
    print(df)

    ds = df.to_xarray()
    ds.to_netcdf('my_3d_dataarray.nc')


# %%
if __name__ == "__main__":
    process_data()


