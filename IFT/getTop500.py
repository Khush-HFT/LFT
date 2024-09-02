# import fmpsdk
# import pandas as pd
# api_key = "USpJqLtmwNP3brqw33DQGEramLU4VvSx"

# df = pd.read_csv('allCompanies.csv')

# map = {}
# count = 0
# for symbol in df['Symbol']:
#     count += 1
#     if count > 10:
#         break
#     newSymbol = symbol + '.NS'
#     data = fmpsdk.market_capitalization(apikey=api_key, symbol=newSymbol)
#     data = pd.DataFrame(data)
#     map[symbol] = data['marketCap'].values[0]

# sorted_map = dict(sorted(map.items(), key=lambda item: item[1], reverse=True))

# df = pd.DataFrame(list(sorted_map.items()), columns=['Symbol', 'marketCap'])

# # Get the top 10 values
# top_500_df = df.head(5)

# # Define the CSV file name
# csv_file_name = 'ind_nifty500list.csv'

# # Write to CSV
# top_500_df.to_csv(csv_file_name, index=False)



import fmpsdk
import pandas as pd

def get_top_n_stocks_by_market_cap(api_key, input_csv, output_csv, sort_by, n=500):
    df = pd.read_csv(input_csv)

    market_cap_map = {}

    for count, symbol in enumerate(df['Symbol'], start=1):
        if count > n:
            break
        new_symbol = symbol + '.NS'
        data = fmpsdk.market_capitalization(apikey=api_key, symbol=new_symbol)
        data = pd.DataFrame(data)
        market_cap_map[symbol] = data[sort_by].values[0]

    sorted_market_cap_map = dict(sorted(market_cap_map.items(), key=lambda item: item[1], reverse=True))

    top_n_df = pd.DataFrame(list(sorted_market_cap_map.items()), columns=['Symbol', sort_by])

    top_n_df.to_csv(output_csv, index=False)

# Example usage
api_key = "USpJqLtmwNP3brqw33DQGEramLU4VvSx"
input_csv = 'allCompanies.csv'
output_csv = 'ind_nifty500list.csv'
sort_by = 'marketCap'
get_top_n_stocks_by_market_cap(api_key, input_csv, output_csv, sort_by, n=50)