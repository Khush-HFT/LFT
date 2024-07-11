import pandas as pd
import numpy as np
import xarray as xr
from threeDF import DataLoader
import datetime as dt
from scipy.stats import rankdata

class FundamentalData:
    def __init__(self, dates_file, stocks_file, data_file):
        self.date_df = pd.read_csv(dates_file)
        self.stocks_df = pd.read_csv(stocks_file)
        self.data = xr.open_dataset(data_file)

    @property
    def rps(self):
        return self.data['revenuePerShare'].to_dataframe()

    @property
    def market_cap(self):
        return self.data['marketCap'].to_dataframe()
    
    @property
    def close(self):
        return self.data['close'].to_dataframe()
    
    @property
    def eps(self):
        return self.data['eps'].to_dataframe()
    
   

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
    return rank(delta(fundamental_data.rps)) / fundamental_data.market_cap



if __name__ == "__main__":
    obj = FundamentalData('date.csv', 'ind_nifty500list.csv', 'my_3d_dataarray.nc')
    # print(adx(obj.data, 14))
    # print(vwap(obj.data, 1))
    print(obj.eps[:50])