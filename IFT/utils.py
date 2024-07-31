import pandas as pd
import numpy as np
import xarray as xr
from threeDF import DataLoader
import datetime as dt
from scipy.stats import rankdata
import itertools
import matplotlib.pyplot as plt
import re



class FundamentalData:
    def __init__(self, dates_file, stocks_file, data_file, sector_file):
        self.date_df = pd.read_csv(dates_file)
        self.stocks_df = pd.read_csv(stocks_file)
        self.data = xr.open_dataset(data_file)
        self.sectorData = pd.read_csv(sector_file)
    

    @property
    def market_cap(self):
        return self.data['marketCap'].to_dataframe(), 'marketCap'
    
    @property
    def close(self):
        return self.data['close'].to_dataframe(), 'close'

    @property
    def eps(self):
        return self.data['eps'].to_dataframe(), 'eps'

    @property
    def npm(self):
        return self.data['netProfitMargin'].to_dataframe(), 'netProfitMargin'

    @property
    def cr(self):
        return self.data['currentRatio'].to_dataframe(), 'currentRatio'

    @property
    def per(self):
        return self.data['peRatio'].to_dataframe(), 'peRatio'

    @property
    def ptsr(self):
        return self.data['priceToSalesRatio'].to_dataframe(), 'priceToSalesRatio'

    @property
    def pocfr(self):
        return self.data['pocfratio'].to_dataframe(), 'pocfratio'

    @property
    def pfcf(self):
        return self.data['pfcfRatio'].to_dataframe(), 'pfcfRatio'

    @property
    def pbr(self):
        return self.data['pbRatio'].to_dataframe(), 'pbRatio'

    @property
    def ev_to_sales(self):
        return self.data['evToSales'].to_dataframe(), 'evToSales'

    @property
    def enterprise_value_over_ebitda(self):
        return self.data['enterpriseValueOverEBITDA'].to_dataframe(), 'enterpriseValueOverEBITDA'

    @property
    def ev_to_operating_cash_flow(self):
        return self.data['evToOperatingCashFlow'].to_dataframe(), 'evToOperatingCashFlow'

    @property
    def earnings_yield(self):
        return self.data['earningsYield'].to_dataframe(), 'earningsYield'

    @property
    def free_cash_flow_yield(self):
        return self.data['freeCashFlowYield'].to_dataframe(), 'freeCashFlowYield'

    @property
    def dte(self):
        return self.data['debtToEquity'].to_dataframe(), 'debtToEquity'

    @property
    def debt_to_assets(self):
        return self.data['debtToAssets'].to_dataframe(), 'debtToAssets'

    @property
    def net_debt_to_ebitda(self):
        return self.data['netDebtToEBITDA'].to_dataframe(), 'netDebtToEBITDA'

    @property
    def interest_coverage(self):
        return self.data['interestCoverage'].to_dataframe(), 'interestCoverage'

    @property
    def income_quality(self):
        return self.data['incomeQuality'].to_dataframe(), 'incomeQuality'

    @property
    def dividend_yield(self):
        return self.data['dividendYield'].to_dataframe(), 'dividendYield'

    @property
    def payout_ratio(self):
        return self.data['payoutRatio'].to_dataframe(), 'payoutRatio'

    @property
    def sales_general_and_administrative_to_revenue(self):
        return self.data['salesGeneralAndAdministrativeToRevenue'].to_dataframe(), 'salesGeneralAndAdministrativeToRevenue'

    @property
    def return_on_tangible_assets(self):
        return self.data['returnOnTangibleAssets'].to_dataframe(), 'returnOnTangibleAssets'

    @property
    def working_capital(self):
        return self.data['workingCapital'].to_dataframe(), 'workingCapital'

    @property
    def tangible_asset_value(self):
        return self.data['tangibleAssetValue'].to_dataframe(), 'tangibleAssetValue'

    @property
    def net_current_asset_value(self):
        return self.data['netCurrentAssetValue'].to_dataframe(), 'netCurrentAssetValue'

    @property
    def average_receivables(self):
        return self.data['averageReceivables'].to_dataframe(), 'averageReceivables'

    @property
    def receivables_turnover(self):
        return self.data['receivablesTurnover'].to_dataframe(), 'receivablesTurnover'

    @property
    def rps(self):
        return self.data['revenuePerShare'].to_dataframe(), 'revenuePerShare'

    @property
    def capex_per_share(self):
        return self.data['capexPerShare'].to_dataframe(), 'capexPerShare'

    @property
    def quick_ratio(self):
        return self.data['quickRatio'].to_dataframe(), 'quickRatio'

    @property
    def cash_ratio(self):
        return self.data['cashRatio'].to_dataframe(), 'cashRatio'

    @property
    def gross_profit_margin(self):
        return self.data['grossProfitMargin'].to_dataframe(), 'grossProfitMargin'

    @property
    def return_on_assets(self):
        return self.data['returnOnAssets'].to_dataframe(), 'returnOnAssets'

    @property
    def return_on_capital_employed(self):
        return self.data['returnOnCapitalEmployed'].to_dataframe(), 'returnOnCapitalEmployed'

    @property
    def company_equity_multiplier(self):
        return self.data['companyEquityMultiplier'].to_dataframe(), 'companyEquityMultiplier'

    @property
    def net_income_per_ebt(self):
        return self.data['netIncomePerEBT'].to_dataframe(), 'netIncomePerEBT'

    @property
    def long_term_debt_to_capitalization(self):
        return self.data['longTermDebtToCapitalization'].to_dataframe(), 'longTermDebtToCapitalization'

    @property
    def total_debt_to_capitalization(self):
        return self.data['totalDebtToCapitalization'].to_dataframe(), 'totalDebtToCapitalization'

    @property
    def fixed_asset_turnover(self):
        return self.data['fixedAssetTurnover'].to_dataframe(), 'fixedAssetTurnover'

    @property
    def operating_cash_flow_per_share(self):
        return self.data['operatingCashFlowPerShare'].to_dataframe(), 'operatingCashFlowPerShare'

    @property
    def free_cash_flow_per_share(self):
        return self.data['freeCashFlowPerShare'].to_dataframe(), 'freeCashFlowPerShare'

    @property
    def cash_flow_coverage_ratios(self):
        return self.data['cashFlowCoverageRatios'].to_dataframe(), 'cashFlowCoverageRatios'

    @property
    def short_term_coverage_ratios(self):
        return self.data['shortTermCoverageRatios'].to_dataframe(), 'shortTermCoverageRatios'

    @property
    def capital_expenditure_coverage_ratio(self):
        return self.data['capitalExpenditureCoverageRatio'].to_dataframe(), 'capitalExpenditureCoverageRatio'

    @property
    def dividend_paid_and_capex_coverage_ratio(self):
        return self.data['dividendPaidAndCapexCoverageRatio'].to_dataframe(), 'dividendPaidAndCapexCoverageRatio'

    @property
    def days_of_sales_outstanding(self):
        return self.data['daysOfSalesOutstanding'].to_dataframe(), 'daysOfSalesOutstanding'

    @property
    def days_of_inventory_outstanding(self):
        return self.data['daysOfInventoryOutstanding'].to_dataframe(), 'daysOfInventoryOutstanding'

    @property
    def operating_cycle(self):
        return self.data['operatingCycle'].to_dataframe(), 'operatingCycle'

    @property
    def days_of_payables_outstanding(self):
        return self.data['daysOfPayablesOutstanding'].to_dataframe(), 'daysOfPayablesOutstanding'

    @property
    def cash_conversion_cycle(self):
        return self.data['cashConversionCycle'].to_dataframe(), 'cashConversionCycle'

    @property
    def operating_profit_margin(self):
        return self.data['operatingProfitMargin'].to_dataframe(), 'operatingProfitMargin'

    @property
    def pretax_profit_margin(self):
        return self.data['pretaxProfitMargin'].to_dataframe(), 'pretaxProfitMargin'

    @property
    def net_profit_margin(self):
        return self.data['netProfitMargin'].to_dataframe(), 'netProfitMargin'

    @property
    def effective_tax_rate(self):
        return self.data['effectiveTaxRate'].to_dataframe(), 'effectiveTaxRate'

    @property
    def net_income_per_ebt(self):
        return self.data['netIncomePerEBT'].to_dataframe(), 'netIncomePerEBT'

    @property
    def ebt_per_ebit(self):
        return self.data['ebtPerEbit'].to_dataframe(), 'ebtPerEbit'

    @property
    def debt_ratio(self):
        return self.data['debtRatio'].to_dataframe(), 'debtRatio'

    @property
    def debt_equity_ratio(self):
        return self.data['debtEquityRatio'].to_dataframe(), 'debtEquityRatio'

    @property
    def cash_flow_to_debt_ratio(self):
        return self.data['cashFlowToDebtRatio'].to_dataframe(), 'cashFlowToDebtRatio'

    @property
    def asset_turnover(self):
        return self.data['assetTurnover'].to_dataframe(), 'assetTurnover'

    @property
    def capital_expenditure_coverage_ratio(self):
        return self.data['capitalExpenditureCoverageRatio'].to_dataframe(), 'capitalExpenditureCoverageRatio'

    @property
    def price_earnings_to_growth_ratio(self):
        return self.data['priceEarningsToGrowthRatio'].to_dataframe(), 'priceEarningsToGrowthRatio'

    @property
    def enterprise_value_multiple(self):
        return self.data['enterpriseValueMultiple'].to_dataframe(), 'enterpriseValueMultiple'

    @property
    def price_fair_value(self):
        return self.data['priceFairValue'].to_dataframe(), 'priceFairValue'

    @property
    def return_on_capital_employed(self):
        return self.data['returnOnCapitalEmployed'].to_dataframe(), 'returnOnCapitalEmployed'

class FinancialDataProcessor:
    def __init__(self, dates_file, stocks_file, data_file, sector_file, alpha_function):
        self.fundamental_data = FundamentalData(dates_file, stocks_file, data_file, sector_file)
        self.alpha_function = alpha_function
        self.dates = self.fundamental_data.date_df
        self.stocks = self.fundamental_data.stocks_df
        self.data = self.fundamental_data.data.to_dataframe()
        self.weight_matrix = np.zeros((len(self.dates), len(self.stocks)))
        self.normalized_weight_matrix = None
        self.pnl_matrix = np.zeros((len(self.dates), len(self.stocks)))
        self.cumulative_pnl_matrix = np.zeros((len(self.dates), len(self.stocks)))
        self.all_sectors = self.fundamental_data.sectorData['sector'].unique()
        self.sector_stocks = []
        self.values = {
            'fundamental_data.npm': self.fundamental_data.npm,
            'fundamental_data.dte': self.fundamental_data.dte,
            'fundamental_data.netProfitMargin': self.fundamental_data.net_profit_margin,
            'fundamental_data.revenuePerShare': self.fundamental_data.rps,
            'fundamental_data.operatingProfitMargin': self.fundamental_data.operating_profit_margin,
            'fundamental_data.pretaxProfitMargin': self.fundamental_data.pretax_profit_margin,
            'fundamental_data.effectiveTaxRate': self.fundamental_data.effective_tax_rate,
            'fundamental_data.netIncomePerEBT': self.fundamental_data.net_income_per_ebt,
            'fundamental_data.ebtPerEbit': self.fundamental_data.ebt_per_ebit,
            'fundamental_data.debtRatio': self.fundamental_data.debt_ratio,
            'fundamental_data.debtEquityRatio': self.fundamental_data.debt_equity_ratio,
            'fundamental_data.cashFlowToDebtRatio': self.fundamental_data.cash_flow_to_debt_ratio,
            'fundamental_data.assetTurnover': self.fundamental_data.asset_turnover,
            'fundamental_data.capitalExpenditureCoverageRatio': self.fundamental_data.capital_expenditure_coverage_ratio,
            'fundamental_data.priceEarningsToGrowthRatio': self.fundamental_data.price_earnings_to_growth_ratio,
            'fundamental_data.enterpriseValueMultiple': self.fundamental_data.enterprise_value_multiple,
            'fundamental_data.priceFairValue': self.fundamental_data.price_fair_value,
            'fundamental_data.returnOnCapitalEmployed': self.fundamental_data.return_on_capital_employed,
            'fundamental_data.marketCap': self.fundamental_data.market_cap,
            'fundamental_data.close': self.fundamental_data.close,
            'fundamental_data.cr': self.fundamental_data.cr,
        }


    def calculate_weights(self, sector_name):
        alpha_string = self.alpha_function(self.fundamental_data)
        tokens = tokenize(alpha_string)
        postfix_tokens = infix_to_postfix(tokens)
        alpha_values, alpha_name = evaluate_postfix(postfix_tokens, self.values)

        if sector_name == "All":

            for i, j in itertools.product(range(len(self.dates)), range(len(self.stocks))):
                self.weight_matrix[i, j] = alpha_values.loc[self.dates['date'][i], self.stocks['Symbol'][j]]

        else:

            if sector_name not in self.all_sectors:
                raise ValueError("Sector not found")
            
            self.sector_stocks = [
                self.fundamental_data.sectorData['symbol'][i]
                for i in range(len(self.fundamental_data.sectorData))
                if self.fundamental_data.sectorData['sector'][i] == sector_name
            ]
            self.weight_matrix = np.zeros((len(self.dates), len(self.sector_stocks)))

            for i in range(len(self.dates)):
                k = 0
                for j in range(len(self.stocks)):
                    check_stock = self.stocks['Symbol'][j] + '.NS'
                    if check_stock in self.sector_stocks:
                        self.weight_matrix[i, k] = alpha_values.loc[self.dates['date'][i], self.stocks['Symbol'][j]]
                        k += 1

        weights = self.weight_matrix.flatten()
        normalized_weights = weights - np.mean(weights)
        total_abs_sum = np.sum(np.abs(normalized_weights))
        adjustment_factor = 250 / total_abs_sum
        adjusted_weights = normalized_weights * adjustment_factor
        self.normalized_weight_matrix = adjusted_weights.reshape(self.weight_matrix.shape)
        print(self.all_sectors)


    def calculate_pnl(self, sector_name):
        
        if sector_name == "All":
            for i, j in itertools.product(range(len(self.dates)), range(len(self.stocks))):
                daily_pnl = (self.data.at[(self.dates['date'][i], self.stocks['Symbol'][j]), 'close'] - 
                            self.data.at[(self.dates['date'][i], self.stocks['Symbol'][j]), 'open'])
                daily_pnl *= self.normalized_weight_matrix[i, j]
                self.pnl_matrix[i, j] = daily_pnl
        else:
            if sector_name not in self.all_sectors:
                raise ValueError("Sector not found")
            self.pnl_matrix = np.zeros((len(self.dates), len(self.weight_matrix[0])))
            self.cumulative_pnl_matrix = np.zeros((len(self.dates), len(self.weight_matrix[0])))

            for i in range(len(self.dates)):
                k = 0
                for j in range(len(self.stocks)):
                    check_stock = self.stocks['Symbol'][j] + '.NS'
                    if check_stock in self.sector_stocks:
                        daily_pnl = (self.data.at[(self.dates['date'][i], self.stocks['Symbol'][j]), 'close'] - 
                            self.data.at[(self.dates['date'][i], self.stocks['Symbol'][j]), 'open'])
                        daily_pnl *= self.normalized_weight_matrix[i, k]
                        self.pnl_matrix[i, k] = daily_pnl
                        k += 1


        for i in range(len(self.dates)):
            if i == 0:
                self.cumulative_pnl_matrix[i, :] = self.pnl_matrix[i, :]
            else:
                self.cumulative_pnl_matrix[i, :] = self.cumulative_pnl_matrix[i-1, :] + self.pnl_matrix[i, :]
        print(self.cumulative_pnl_matrix)
        return self.cumulative_pnl_matrix

    def plot_daily_pnl(self, sector_name):
        if sector_name == "All":
            pnl_df = pd.DataFrame(self.pnl_matrix, index=self.dates['date'], columns=self.stocks['Symbol'])
        else:   
            pnl_df = pd.DataFrame(self.pnl_matrix, index=self.dates['date'], columns=self.sector_stocks)

        daily_pnl = pnl_df.sum(axis=1)
        daily_pnl.plot()
        plt.xlabel('Date')
        plt.ylabel('PnL')
        plt.show()

    def plot_cumulative_pnl(self, sector_name):
        if sector_name == "All":
            cumulative_pnl_df = pd.DataFrame(self.cumulative_pnl_matrix, index=self.dates['date'], columns=self.stocks['Symbol'])
        else:
            cumulative_pnl_df = pd.DataFrame(self.cumulative_pnl_matrix, index=self.dates['date'], columns=self.sector_stocks)
        cumulative_pnl = cumulative_pnl_df.sum(axis=1)
        cumulative_pnl.plot()
        plt.xlabel('Date')
        plt.ylabel('Cumulative PnL')
        plt.title('Cumulative PnL Over Time')
        plt.show()
    

def tokenize(expression):
    return re.findall(r'[a-zA-Z_]\w*(?:\.\w+)*|[\+\-\*/()]', expression)

def infix_to_postfix(tokens):
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2}
    output = []
    operators = []
    
    for token in tokens:
        if re.match(r'[a-zA-Z_]\w*(?:\.\w+)*', token):
            output.append(token)
        elif token in precedence:
            while (operators and operators[-1] in precedence and
                   precedence[token] <= precedence[operators[-1]]):
                output.append(operators.pop())
            operators.append(token)
        elif token == '(':
            operators.append(token)
        elif token == ')':
            while operators and operators[-1] != '(':
                output.append(operators.pop())
            operators.pop()
    
    while operators:
        output.append(operators.pop())
    
    return output

def evaluate_postfix(postfix_tokens, values):
    # sourcery skip: avoid-builtin-shadow
    stack = []
    for token in postfix_tokens:
        if re.match(r'[a-zA-Z_]\w*(?:\.\w+)*', token):
            stack.append(values[token])
        else:
            b = stack.pop()
            a = stack.pop()
            if token == '+':
                stack.append(add(a, b))
            elif token == '-':
                stack.append(subtract(a, b))
            elif token == '*':
                stack.append(multiply(a, b))
            elif token == '/':
                stack.append(divide(a, b))
    return stack[0]


def add (fundamental_data1, fundamental_data2):
    # sourcery skip: avoid-builtin-shadow
    data1, fundamental_data1_name = fundamental_data1
    data2, fundamental_data2_name = fundamental_data2

    new_data = data1.copy()
    new_data.rename(columns={fundamental_data1_name: 'alpha_function_result'}, inplace=True)

    for company in new_data.index.get_level_values('company').unique():
        for date in new_data.index.get_level_values('date').unique():
            sum = data1.loc[(date, company), fundamental_data1_name] + data2.loc[(date, company), fundamental_data2_name]
            new_data.loc[(date, company), 'alpha_function_result'] = sum

    return new_data, 'alpha_function_result'

def subtract (fundamental_data1, fundamental_data2):
    # sourcery skip: avoid-builtin-shadow
    data1, fundamental_data1_name = fundamental_data1
    data2, fundamental_data2_name = fundamental_data2

    new_data = data1.copy()
    new_data.rename(columns={fundamental_data1_name: 'alpha_function_result'}, inplace=True)

    for company in new_data.index.get_level_values('company').unique():
        for date in new_data.index.get_level_values('date').unique():
            diff = data1.loc[(date, company), fundamental_data1_name] - data2.loc[(date, company), fundamental_data2_name]
            new_data.loc[(date, company), 'alpha_function_result'] = diff

    return new_data, 'alpha_function_result'

def multiply (fundamental_data1, fundamental_data2):
    # sourcery skip: avoid-builtin-shadow
    data1, fundamental_data1_name = fundamental_data1
    data2, fundamental_data2_name = fundamental_data2

    new_data = data1.copy()
    new_data.rename(columns={fundamental_data1_name: 'alpha_function_result'}, inplace=True)

    for company in new_data.index.get_level_values('company').unique():
        for date in new_data.index.get_level_values('date').unique():
            product = data1.loc[(date, company), fundamental_data1_name] * data2.loc[(date, company), fundamental_data2_name]
            new_data.loc[(date, company), 'alpha_function_result'] = product

    return new_data, 'alpha_function_result'

def divide (fundamental_data1, fundamental_data2):
    # sourcery skip: avoid-builtin-shadow
    data1, fundamental_data1_name = fundamental_data1
    data2, fundamental_data2_name = fundamental_data2

    new_data = data1.copy()
    new_data.rename(columns={fundamental_data1_name: 'alpha_function_result'}, inplace=True)

    for company in new_data.index.get_level_values('company').unique():
        for date in new_data.index.get_level_values('date').unique():
            quotient = data1.loc[(date, company), fundamental_data1_name] / data2.loc[(date, company), fundamental_data2_name]
            new_data.loc[(date, company), 'alpha_function_result'] = quotient

    return new_data, 'alpha_function_result'

def delta(data):
    for time_value in data.index.get_level_values('date').unique():
        for company_value in data.index.get_level_values('company').unique():
            # time_value = pd.to_datetime(time_value)
            new_date = DataLoader.get_first_date_of_quarter(time_value)
            pre_date = DataLoader.get_first_date_of_quarter(new_date - dt.timedelta(days=1))
            while new_date not in data.index.get_level_values('date').unique():
                new_date = new_date + dt.timedelta(days=1)
            while pre_date not in data.index.get_level_values('date').unique():
                pre_date = pre_date + dt.timedelta(days=1)
            data.loc[time_value, company_value] = data.loc[new_date, company_value] - data.loc[pre_date, company_value]
    return data

def rank(data_array):
    def rescale_ranks(values):
        n = len(values)
        ranks = rankdata(values, method='average')
        rescaled_ranks = 2 * ((ranks - 1) / (n - 1)) - 1
        return rescaled_ranks

    ranked_data = data_array.copy()
    for date in data_array.index.get_level_values('date').unique():
        # Handling each column separately to maintain the DataFrame structure
        ranked_values = np.apply_along_axis(rescale_ranks, 0, data_array.loc[date].values)
        ranked_data.loc[date] = ranked_values

    return ranked_data

def ts_rank(data_array, t: int):
    data_array = data_array[:t*len(data_array.index.get_level_values('company').unique())]

    def normalize_rank(rank):
        return (rank - 1) / (t - 1)

    for company in data_array.index.get_level_values('company').unique():
        array = []
        for date in data_array.index.get_level_values('date').unique():
            for value in data_array.loc[date, company].values:
                array.append(value)
        ranks = rankdata(array, method='average')
        ranks = np.apply_along_axis(normalize_rank, 0, ranks)

        ind = 0
        for date in data_array.index.get_level_values('date').unique():
            data_array.loc[date, company] = ranks[ind]
            ind += 1
    return data_array

def ema(x, decay : float):
    def iterEma(x, decay):
        emaData = []
        for ind in range(len(x)):
            if ind == 0:
                emaData.append(x[ind])
            else:
                emaData.append(decay * emaData[ind-1] + (1 - decay) * x[ind])
        return emaData
    
    for company in x.index.get_level_values('company').unique():
        array = []
        for date in x.index.get_level_values('date').unique():
            array.extend(iter(x.loc[date, company].values))
        array.reverse()

        emaData = iterEma(array, decay)

        emaData.reverse()
        i = 0
        for date in x.index.get_level_values('date').unique():
            x.loc[(date, company), 'close'] = emaData[i]
            i += 1
    return x

def rsi(data, period):
    for company in data.index.get_level_values('company').unique():
        array = []
        rsi_series = []

        for date in data.index.get_level_values('date').unique():
            array.extend(iter(data.loc[date, company].values))

        array.reverse()

        for i in range(period, len(array)):
            closing_data = array[i - period:i]
            deltas = np.diff(closing_data)
            up_vals = deltas[deltas >= 0].sum() / period
            low_vals = -deltas[deltas < 0].sum() / period

            rs = up_vals / low_vals
            rsi = 100.0 - (100.0 / (1.0 + rs))  # RSI formula
            rsi_series.append(rsi)

        rsi_series.reverse()

        for ind, date in enumerate(data.index.get_level_values('date').unique()):
            data.loc[date, company] = (
                np.nan if (ind >= len(rsi_series)) else rsi_series[ind]
            )
    return data

def adx(data, period):
    def DM(high, low):
        posDM = [-1]
        negDM = [-1]
        for i in range(1, len(high)):
            upMove = high[i] - high[i - 1]
            downMove = low[i - 1] - low[i]
            posDM.append(upMove if (upMove > downMove and upMove > 0) else 0)
            negDM.append(downMove if (downMove > upMove and downMove > 0) else 0)
    
        return posDM, negDM
    
    def TR(high, low, close):
        TR = [-1]
        for i in range(1, len(high)):
            TR.append(max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1])))
        return TR
    
    def smoothVal(TR, period):
        smoothedVal = [np.nan] * period
        for i in range(period, len(TR)):
            smoothedVal.append(np.sum(TR[i - period:i]) / period)
        return smoothedVal
    
    def DI(smoothPosDM, smoothNegDM, smoothTR):
        DIplus = []
        DIneg = []
        for i in range(len(smoothTR)):
            DIplus.append((smoothPosDM[i] / smoothTR[i]) * 100)
            DIneg.append((smoothNegDM[i] / smoothTR[i]) * 100)

        return DIplus, DIneg
    
    def DX(DIplus, DIneg):
        DX = []
        for i in range(len(DIplus)):
            DX.append(abs(DIplus[i] - DIneg[i]) / (DIplus[i] + DIneg[i]) * 100)
        return DX
    
    def ADXval(DX, period, index):
        if index == len(DX):
            return

        if index == period:
            ADXvals.append(np.sum(DX[-period:]) / period)
        else:
            ADXvals.append((ADXvals[index -1] * (period - 1) + DX[index]) / period)

        ADXval(DX, period, index + 1)

        
    
    data = data.to_dataframe()
    for company in data.index.get_level_values('company').unique():
        high = []
        low = []
        close = []
        ADXvals = [np.nan] * period
        for date in data.index.get_level_values('date').unique():
            high.append(data.loc[(date, company), 'high'])
            low.append(data.loc[(date, company), 'low'])
            close.append(data.loc[(date, company), 'close'])
        high.reverse()
        low.reverse()
        close.reverse()
        
        posDM, negDM = DM(high, low)
        TRs = TR(high, low, close)
        smoothTRs = smoothVal(TRs, period)
        smoothPosDM = smoothVal(posDM, period)
        smoothNegDM = smoothVal(negDM, period)

        DIplus, DIneg = DI(smoothPosDM, smoothNegDM, smoothTRs)

        DXval = DX(DIplus, DIneg)
        ADXval(DXval, period, period)

        ADXvals.reverse()
        ind = 0
        for date in data.index.get_level_values('date').unique():
            data.loc[(date, company), 'adx'] = ADXvals[ind]
            ind += 1

    return data
    
def macd(data, short_period, long_period):
    def emaPeriod(data, period, ind):
        emaData = []
        sum = 0
        for i in range(len(data)):
            if i < period-1:
                emaData.append(np.nan)
                sum += data[i]
            elif i == period-1:
                emaData.append(sum / period)
            else:
                emaData.append((data[ind]*(2/period+1)) + emaData[ind-1]*(1-(2/period+1)))
        return emaData
    
    for company in data.index.get_level_values('company').unique():
        array = []
        for date in data.index.get_level_values('date').unique():
            array.extend(iter(data.loc[date, company].values))
        array.reverse()
        
        short_ema = emaPeriod(array, short_period, 0)
        long_ema = emaPeriod(array, long_period, 0)
        macd_line = [short_ema[i] - long_ema[i] for i in range(len(short_ema))]
        macd_line.reverse()
        i = 0
        for date in data.index.get_level_values('date').unique():
            data.loc[date, company] = macd_line[i]
            i+=1
    return data

def vwap(data, period):
    data = data.to_dataframe()
    for company in data.index.get_level_values('company').unique():
        volume = []
        price = []
        for date in data.index.get_level_values('date').unique():
            data.loc[(date, company), 'vwap'] = np.nan
            volume.append(data.loc[(date, company), 'volume'])
            price.append(data.loc[(date, company), 'close'])
        volume.reverse()
        price.reverse()
        vwap = []
        priceSum = 0
        volumeSum = 0
        for i in range(len(volume)):
            if i < period-1:
                priceSum += price[i]
                volumeSum += volume[i]
                vwap.append(np.nan)
            else:
                priceSum += price[i]
                volumeSum += volume[i]
                if i != 0:
                    priceSum -= price[i-period]
                    volumeSum -= volume[i-period]
                vwap.append((priceSum*volumeSum) / volumeSum)

        for date in data.index.get_level_values('date').unique():
            data.loc[(date, company), 'vwap'] = vwap.pop()
                    
    return data

def alpha_example_1(fundamental_data):
    return fundamental_data.rps

def alpha_example_2(fundamental_data):
    return delta(fundamental_data.rps) / fundamental_data.market_cap

def alpha_example_3(fundamental_data):
    data =  rank(delta(fundamental_data.rps)) 

    market_cap_data = fundamental_data.market_cap
    for company in data.index.get_level_values('company').unique():
        for date in data.index.get_level_values('date').unique():
            if market_cap_data.loc[(date, company), 'marketCap'] == 0:
                data.loc[(date, company), 'revenuePerShare'] = np.nan
            else:
                data.loc[(date, company), 'revenuePerShare'] = data.loc[(date, company), 'revenuePerShare'] / market_cap_data.loc[(date, company), 'marketCap']
    data.rename(columns={'revenuePerShare': 'alpha_example_3_result'}, inplace=True)
    return data

def alpha_example_4(fundamental_data):
    fundamental_data_npm = fundamental_data.npm
    fundamental_data_pbr = fundamental_data.pbr
    fundamental_data_dte = fundamental_data.dte
    # sourcery skip: avoid-builtin-shadow
    for company in fundamental_data_npm.index.get_level_values('company').unique():
        for date in fundamental_data_npm.index.get_level_values('date').unique():
            # data.loc[(date, company), 'vwap'] = np.nan
            sum = fundamental_data_npm.loc[(date, company), 'netProfitMargin'] + fundamental_data_pbr.loc[(date, company), 'pbRatio'] + fundamental_data_dte.loc[(date, company), 'debtToEquity']

            fundamental_data_npm.loc[(date, company), 'netProfitMargin'] = sum
    fundamental_data_npm.rename(columns={'netProfitMargin': 'alpha_example_4_result'}, inplace=True)
    return fundamental_data_npm

def alpha_example_5(fundamental_data):
    fundamental_data_npm = fundamental_data.npm
    fundamental_data_pbr = fundamental_data.pbr
    for company in fundamental_data_npm.index.get_level_values('company').unique():
        for date in fundamental_data_npm.index.get_level_values('date').unique():
            sum = fundamental_data_npm.loc[(date, company), 'netProfitMargin'] + fundamental_data_pbr.loc[(date, company), 'pbRatio']
            fundamental_data_npm.loc[(date, company), 'netProfitMargin'] = sum
    fundamental_data_npm.rename(columns={'netProfitMargin': 'alpha_example_5_result'}, inplace=True)
    return fundamental_data_npm


if __name__ == "__main__":
    obj = FundamentalData('date.csv', 'ind_nifty500list.csv', 'my_3d_dataarray.nc', 'sectorData.csv')
    obj2 = FinancialDataProcessor('date.csv', 'ind_nifty500list.csv', 'my_3d_dataarray.nc', 'sectorData.csv', alpha_example_1)

