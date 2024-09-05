import fmpsdk
import pandas as pd
import requests

api_key = "USpJqLtmwNP3brqw33DQGEramLU4VvSx"

df = pd.read_csv('allCompanies.csv')

map = {}
count = 0
for symbol in df['Symbol']:
    count += 1
    if count > 10:
        break
    newSymbol = symbol + '.NS'
    data = fmpsdk.market_capitalization(apikey=api_key, symbol=newSymbol)

    url = f"https://financialmodelingprep.com/api/v3/income-statement/{newSymbol}?period=annual&apikey=USpJqLtmwNP3brqw33DQGEramLU4VvSx"

    payload={}
    headers = {
    'Accept': 'application/json'
    }

    response = requests.request("GET", url, headers=headers, data=payload).json()    
    # Convert to DataFrame
    data = pd.DataFrame(data)
    data2 = pd.DataFrame(response)
    
    # Get market cap and EBITDA
    market_cap = data['marketCap'].values[0] if 'marketCap' in data.columns and len(data) > 0 else None
    ebitda = data2['ebitda'].values[0] if 'ebitda' in data2.columns and len(data2) > 0 else None
    # print(ebitda, newSymbol)
    # print(data2['ebitda'].values[0], newSymbol)
    
    # Store in map
    map[symbol] = {'marketCap': market_cap, 'ebitda': ebitda}

# Convert map to DataFrame
df = pd.DataFrame.from_dict(map, orient='index').reset_index()
df.columns = ['Symbol', 'marketCap', 'ebitda']

# Get top 10 companies
top_500_df = df.head(10)

csv_file_name = 'ind_nifty500list.csv'
top_500_df.to_csv(csv_file_name, index=False)

print("CSV saved as:", csv_file_name)



