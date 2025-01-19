import pandas as pd

# csv_filepath = '/home/elliotg/pythoning/py_learning/pandas_learning/pokemon_data.csv'
csv_filepath = '/root/py_learning/pandas_learning/pokemon_data.csv'

df = pd.read_csv(csv_filepath)

all_columns = df.columns
name_column = df['Type 1'][0:5] # first 5 of type 1
list_of_columns = df[['Name', 'Type 1', 'Type 2']] 
nth_row = df.iloc[0:4] # specific rows (0-4th)

 # read a specific location
 
specific_data_point = df.iloc[2, 1]

# iterate over all rows

# for index, row in df.iterrows():
#     print(index, row[['Name', 'Type 1']])

# finding specific data in dataset (not index based)

df.loc[df['Type 1'] == 'Fire']


#high level overview of data

high_level_data = df.describe()

#sorting values

alphabetical_orders = df.sort_values('Name')

# sort by multiple values

mx = df.sort_values(['Type 1', 'HP'], ascending=[1,0]) # 1st is acesnding and 2nd is descending


# making changes to the dataframe


# create a total catagory 

# df['Total'] = df['HP'] + df['Attack'] + df['Defense'] + df['Sp. Atk'] + df['Sp. Def'] + df['Speed']

df['Total'] = df.iloc[:, 4:10].sum(axis=1) # cleaner way of adding columns, axis 1 = horizontally, we added columns from 4 to 9

cols = list(df.columns)

# sort by highest total

best_char = df.sort_values(['Total'], ascending=False)

# reording the columns

df = df[cols[0:4] + [cols[-1]] + cols[4:12]]

#dropping a column

# df = df.drop(columns=['Total'])

# create a csv

# df.to_csv('modified.csv', index=False) # creates a new csv with new ordered columns

#----------------------------------

# filtering the data
# and = &, or = |

import re

# new_df = df.loc[(df['Type 1'] == 'Grass') & (df['Type 2'] == 'Poison') & (df['HP'] > 65)]
only_mega = df.loc[df['Name'].str.contains('Mega')] # get all pokemon which have 'Mega' in their name
not_mega = df.loc[~df['Name'].str.contains('Mega')] # get all pokemon which DONT have 'Mega' in their name
example_regex = df.loc[df['Type 1'].str.contains('fire|grass', regex=True, flags=re.I)] 
example_2_regex = df.loc[df['Name'].str.contains('^pi[a-z]*', regex=True, flags=re.I)]

# df.loc[df['Type 1'] == 'Fire', 'Type 1'] = 'Flamer' # changed column name from Fire to Flamer
# df.loc[df['Type 1'] == 'Fire', 'Legendary']  = True # made it so all flamer pokemon are legendary

only_flamer = df.loc[df['Type 1'] == 'Fire']

# conditional Changes

# df.loc[df['Total'] > 500, ['Generation', 'Legendary']] = ['Test 1', 'Test 2'] # if total is greater than 500, the two columns now will equal TEST VALUE

# Aggregate Statistics

# df.groupby(['Type 1']).mean().sort_values('Defense', ascending=False) # gives all stats based on Type 1, and sorts by Defence 
# df['count'] = 1
# could also do 'count' or 'sum'

# Working with LARGE amounts of data

for df in pd.read_csv('modified.csv', chunksize=1):
    results = df.groupby(['Type 1']).count()
    new_df = pd.concat([new_df, results])




