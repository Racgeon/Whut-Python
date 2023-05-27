import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

country=pd.read_csv('data/GlobalLandTemperaturesByCountry.csv',delimiter=',',encoding='utf-8',index_col='Country')
country.dropna(axis=0,how='any',inplace=True)
groups=country.groupby('Country')
China:pd.DataFrame=groups.get_group('China')
China=China.loc[China['dt']>'1900-00-00']
plt.plot(China['dt'],China['AverageTemperature'])
plt.show()
# print(China['AverageTemperatureUncertainty'])

# Entity - Contains the name of the countries and the regions.

# Code - Information about country code and where code has the value 'Region', it denotes division by grouping various countries.

# Year - Year from 1980-2020

# Cellular Subscription - Mobile phone subscriptions per 100 people.
# This number can get over 100 when the average person has more than one subscription to a mobile service.

# Internet Users(%) - The share of the population that is accessing the internet for all countries of the world.
# No. of Internet Users - Number of people using the Internet in every country.

# Broadband Subscription - The number of fixed broadband subscriptions per 100 people.
# This refers to fixed subscriptions to high-speed access to the public Internet (a TCP/IP connection),
# at downstream speeds equal to, or greater than, 256 kbit/s.