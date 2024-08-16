# this file is used for testing codes


import xarray as xr

# Example 3D DataFrame creation for demonstration
# Assume you have a 3D DataFrame (xarray.Dataset)
data = xr.Dataset(
    {
        'adx': (['date', 'company', 'metric'], [[1, 2], [3, 4]]),
        'other1': (['date', 'company', 'metric'], [[5, 6], [7, 8]]),
        'other2': (['date', 'company', 'metric'], [[9, 10], [11, 12]])
    },
    coords={
        'date': ['2024-01-01', '2024-01-02'],
        'company': ['CompanyA', 'CompanyB'],
        'metric': ['Metric1', 'Metric2']
    }
)

# Print original data
print("Original Data:")
print(data)

# Drop all columns except "adx"
columns_to_keep = ['adx']
data_filtered = data[columns_to_keep]

# Print filtered data
print("\nFiltered Data:")
print(data_filtered)