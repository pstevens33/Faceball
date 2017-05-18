import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn

pd.options.display.max_columns = None
df = pd.read_json('../data/batting_basic.json', lines=True)

# Originally 17 rows without birthday
# All entries were very old without birthdays so dropped them
df = df[df['birthday'].notnull()]

# Deleting entries with heights of '/n    ' and '160lb', 9 total
df = df[(df['height'] != '\n    ') & (df['height'] != '160lb')]

# Get 'height' into useable float format
df['height_feet'] = df['height'].apply(lambda x: x.split('-')[0]).astype('int64')
df['height_inches'] = round(df['height'].apply(lambda x: x.split('-')[1]).astype('int64')/12.0, 3)
df['height'] = df['height_feet'] + df['height_inches']
df.drop(['height_feet', 'height_inches'], axis=1, inplace=True)

# Clean up position columns, strip of commas, delete 58 pitchers that made it through
df['position'] = df['position'].apply(lambda x: x.strip(','))
df = df[df['position'] != 'Pitcher']

# Determine if 'none' in 'last_game' columns are legit
