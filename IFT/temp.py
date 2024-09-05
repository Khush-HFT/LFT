import xarray as xr
import pandas as pd
xarray_3d = xr.open_dataset('my_3d_dataarray.nc')

def filter_companies(xarray_3d, companies_to_keep):
    # Use the list of companies to keep to filter the dataset
    filtered_xarray = xarray_3d.sel(company=companies_to_keep)
    
    return filtered_xarray

# Example usage
companies_to_keep = ['RELIANCE', 'INFY', 'TCS']

# Filter the dataset
new_xarray_3d = filter_companies(xarray_3d, companies_to_keep).to_dataframe()

print(new_xarray_3d)