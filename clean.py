import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

df = pd.read_csv("data/data.csv")
#df.head() # beginning of data
print(df.shape)
drop = []

df['action_type'].unique()
drop.extend(['lon', 'lat', 'game_event_id', 'game_id'])
df['seconds_left'] = 60 * df.loc[:, 'minutes_remaining'] + df.loc[:, 'seconds_remaining']
drop.extend(['minutes_remaining', 'seconds_remaining'])
df['period'].unique()
df['season'] = df['season'].apply(lambda x: x[:4])
df['season'] = pd.to_numeric(df['season']) #keep year only as a number
df['3pt_goal'] = df['shot_type'].str.contains('3PT').astype('int') #is shot a 3
drop.extend(['team_id', 'team_name', 'shot_type', 'shot_zone_range'])
df = df.dropna(subset = ['shot_made_flag']) #remove the null shots bc idk what they're for
df['home_game'] = df['matchup'].str.contains('vs.').astype(int)
drop.append('matchup')

df['shot_distance'] = np.sqrt(df['loc_x']**2 + df['loc_y']**2)

loc_x_zero = df['loc_x'] == 0
df['angle'] = np.array([0]*len(df))
df.loc[~loc_x_zero, 'angle'] = np.arctan(df['loc_y'][~loc_x_zero] / df['loc_x'][~loc_x_zero])
df.loc[loc_x_zero, 'angle'] = np.pi / 2
drop.append('loc_x')
drop.append('loc_y')
#df['angle'][~loc_x_zero] = np.arctan(df['loc_y'][~loc_x_zero] / df['loc_x'][~loc_x_zero])
#df['angle'][loc_x_zero] = np.pi / 2 

#remove date and add year/month/day
df['game_date'] = pd.to_datetime(df['game_date'])
df['game_year'] = df['game_date'].dt.year
df['game_month'] = df['game_date'].dt.month
df['game_day'] = df['game_date'].dt.dayofweek

drop.append('game_date')

df.set_index('shot_id', inplace=True)
#print(df['game_id'].unique().shape)
#print(df['opponent'].unique().shape)
df = df.drop(drop, axis=1)

#move shot made to the end
cols = list(df.columns.values) #Make a list of all of the columns in the df
cols.pop(cols.index('shot_made_flag')) #Create new dataframe with columns in the order you want
df = df[cols+['shot_made_flag']] #Create new dataframe with columns in the order you want

random_sample = df.take(np.random.permutation(len(df))[:3])
#print(random_sample)
print(df.shape)
df.to_csv('data/clean.csv', index=False)
