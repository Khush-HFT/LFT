import pandas as pd
import datetime as dt
import requests
access_token = 'eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiI3TUFQVEEiLCJqdGkiOiI2Njc4YzdhZmJlMzlkYzM0Y2E3MjVjMjAiLCJpc011bHRpQ2xpZW50IjpmYWxzZSwiaWF0IjoxNzE5MTkxNDcxLCJpc3MiOiJ1ZGFwaS1nYXRld2F5LXNlcnZpY2UiLCJleHAiOjE3MTkyNjY0MDB9.c_4h6cHcN2y9UkYG_MzMkHYaunio4g1ipRTDLUZRYcs'

to_date = dt.datetime.now().date() - dt.timedelta(days = 1)

#from date = July 1st 2020
from_date = dt.datetime(2020, 7, 1).date()

requried_data = pd.DataFrame(columns=['date','open','high','low','close','volume','oi'])

url = f"https://api.upstox.com/v2/historical-candle/NSE_EQ|INE002A01018/day/{to_date}/{from_date}"

payload={}
headers = {
'Accept': 'application/json'
}

response = requests.request("GET", url, headers=headers, data=payload).json()
df = pd.DataFrame(response['data']['candles'], columns=["date", "O", "H", "L", "C", "V", "NA"])
df['date'] = pd.to_datetime(df['date'])


df['date'] = df['date'].dt.strftime('%Y-%m-%d')

df.to_csv('date.csv', index=False)
