# Import nflgame, pandas, numpy and matplotlib
import nflgame
import pandas as pd
import numpy as np
import matplotlib as plt

# Initiate season and week variables
week = 1
season = 2017

# Establish games set (games) set referencing week and season variables
games = nflgame.games(season, week=week)

# Establish players set (players) referencing games variables and aggregate data by play
players = nflgame.combine(games, plays=True)

# Initiate dictionary of lists by statistic
dict = {
        'playerName': [],
        'position': [],
        'team': [],
        'playerID': [],
        'rushingAtt': [],
        'rushingYd': [],
        'rushingTD': [],
        'receivingRec': [],
        'receivingYd': [],
        'receivingTD': [],
        'receivingYAC': [],
        'receivingTar': [],
        'passingAtt': [],
        'passingComp': [],
        'passingYd': [],
        'passingTD': [],
        'passingInt': [],
        'defenseInt': [],
        'defenseTkl': [],
        'defenseFFum': [],
        'defenseSk': [],
        'season': [],
        'week': [],
        'gameID': []
        }

# While season less than 2017, execute embedded while loop then increase season by 1
while season <= 2019:

    # While week is less than or equal to week 17, execute embedded for loop then reset week to 1
    while week <= 17:

        # For each player (p) in players set, extract data from games set and append to dictionary
        for p in players:
            dict['playerName'].append(p.name)
            dict['position'].append(p.guess_position)
            dict['team'].append(p.team)
            dict['playerID'].append(p.playerid)
            dict['rushingAtt'].append(p.rushing_att)
            dict['rushingYd'].append(p.rushing_yds)
            dict['rushingTD'].append(p.rushing_tds)
            dict['receivingRec'].append(p.receiving_rec)
            dict['receivingYd'].append(p.receiving_yds)
            dict['receivingTD'].append(p.receiving_tds)
            dict['receivingYAC'].append(p.receiving_yac_yds)
            dict['receivingTar'].append(p.receiving_tar)
            dict['passingAtt'].append(p.passing_att)
            dict['passingComp'].append(p.passing_cmp)
            dict['passingYd'].append(p.passing_yds)
            dict['passingTD'].append(p.passing_tds)
            dict['passingInt'].append(p.passing_int)
            dict['defenseInt'].append(p.defense_int)
            dict['defenseTkl'].append(p.defense_tkl)
            dict['defenseFFum'].append(p.defense_ffum)
            dict['defenseSk'].append(p.defense_sk)
            dict['season'].append(season)
            dict['week'].append(week)
            dict['gameID'].append(str(season) + str(week) + str(p.team))

        week += 1

    season += 1
    week = 1

# Initiate list of columns (names and order) for data frame initiated below
colNames = [
            'playerName',
            'position',
            'team',
            'playerID',
            'rushingAtt',
            'rushingYd',
            'rushingTD',
            'receivingRec',
            'receivingTar',
            'receivingYd',
            'receivingYAC',
            'receivingAirYd',
            'receivingTD',
            'passingAtt',
            'passingComp',
            'passingComp%',
            'passingYd',
            'passingYdAtt',
            'passingTD',
            'passingInt',
            'opptys',
            'teamTotalOpptys',
            'opptyShare',
            'defenseInt',
            'defenseTkl',
            'defenseFFum',
            'defenseSk',
            'season',
            'week',
            'gameID'
            ]

# Convert dictionary (dict) to data frame (index set to Player_name)
df = pd.DataFrame(dict, index=dict['playerName'], columns=colNames)

# Fill im Tyrunn Walker's position (DT, LAR)
df.loc['T.Walker', 'position'] = 'DT'

# Convert gameID column to str (to facilitate indexing by game)
df['gameID'] = df['gameID'].astype(str)

# Calculate air yards by player by week
df['receivingAirYd'] = df['receivingYd'] - df['receivingYAC']

# Calculate passing completion percentage by player by week
df['passingComp%'] = df['passingComp'] / df['passingAtt']

# Calculate passing yards per attempt
df['passingYdAtt'] = df['passingYd'] / df['passingAtt']

# Calculate QB rating by component in new columns ['a', 'b', 'c', 'd'] then drop component columns in final df
df['a'] = (df['passingComp%'] - 0.3) * 5
df['b'] = (df['passingYdAtt'] - 3) * 0.25
df['c'] = (df['passingTD'] / df['passingAtt']) * 20
df['d'] = 2.375 - ((df['passingInt'] / df['passingAtt']) * 25)
df['passingQBR'] = (df['a'] + df['b'] + df['c'] + df['d']) / 6 * 100
#df = df.drop(['a', 'b', 'c', 'd'], axis=1)

# Calculate opportunities by player by week
# Opportunities = rushing attempts + targets
df['opptys'] = df['rushingAtt'] + df['receivingTar']

# Create new data frame (g) set equal to initial data frame (df) grouped by gameID as index
# Calculate total opportunities by team by week and insert column for such into new data frame (df1)
# Total opportunities = total rushing attempts + total targets
g = df.groupby('gameID', as_index=True)
df = df.set_index(['gameID'])
df['teamTotalOpptys'] = g['opptys'].sum()

# Calculate opportunity share by player by week
df['opptyShare'] = df['opptys'] / df['teamTotalOpptys']

# Export data frame (df) to csv
export_csv = df.to_csv('export_dataframe.csv', index=False, header=True)

# Print summary info of data frame (df1)
print df.head(n=70)    # first 70 rows
print df.info()
print df.describe()


# -----------------------------------------------------------------
#  TO DOs:
#  --------
#   Add passer rating, opp team, home tome, away team, tkls for loss, snap counts, kicking /punting, special tms. . .
#   Expected points (EP) per play and winning probability (WP) by play
#   25th, 50th and 75th percentiles won't calculate (at least for rushingYd & rushingAtt) when .describe() printed
#       ignore zero observations in summary stats?