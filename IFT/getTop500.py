import fmpsdk
import pandas as pd
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
    data = pd.DataFrame(data)
    map[symbol] = data['marketCap'].values[0]

sorted_map = dict(sorted(map.items(), key=lambda item: item[1], reverse=True))

df = pd.DataFrame(list(sorted_map.items()), columns=['Symbol', 'marketCap'])

# Get the top 10 values
top_500_df = df.head(5)

# Define the CSV file name
csv_file_name = 'ind_nifty500list.csv'

# Write to CSV
top_500_df.to_csv(csv_file_name, index=False)

