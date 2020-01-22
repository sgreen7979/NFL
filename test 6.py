# import pandas as pd, urllib, numpy as np, and matplotlib.pyplot
import pandas as pd
import urllib
import numpy as np
import matplotlib.pyplot as plt

# Read .csv of play by play data into data frame (pbp)
pbp = pd.read_csv('/Users/samgreen/Desktop/Python/pbp-2019.csv', index_col='playID')

# Change order of columns in pbp
pbp = pbp[['gameID',
           'gameDate',
           'quarter',
           'minute',
           'second',
           'offTeam',
           'defTeam',
           'down',
           'toGo',
           'yardLine',
           'seriesFirstDown',
           'nextScore',
           'description',
           'teamWin',
           'seasonYear',
           'yards',
           'formation',
           'playType',
           'isRush',
           'isPass',
           'isIncomplete',
           'passType',
           'isSack',
           'isChallenge',
           'isChallengeReversed',
           'challenger',
           'isMeasurement',
           'isInterception',
           'isFumble',
           'isPenalty',
           'isTwoPointConversion',
           'isTwoPointConversionSuccessful',
           'rushDirection',
           'yardLineFixed',
           'yardLineDirection',
           'isPenaltyAccepted',
           'penaltyTeam',
           'isNoPlay',
           'penaltyType',
           'penaltyYards']]

# Create a copy of pbp called otherPlays containing all data from pbp for all ex. rush and pass plays
otherPlays = pbp[pbp.playType != 'RUSH']
otherPlays = otherPlays[otherPlays.playType != 'PASS']

# Create a copy of pbp called rushingPlays, which contains all data from pbp for rushing plays only
rushingPlays = pbp[pbp.playType == 'RUSH']

# Extract rushingPlayerName from play description and drop into a new column of rushingPlays
rushingPlays['rushingPlayerName'] = rushingPlays.description.str.extract('\d-(.+?)\s')

# Strip '.' from end of rushingPlayerName elements
rushingPlays.rushingPlayerName = rushingPlays.rushingPlayerName.str.rstrip('.')

# Create a copy of pbp called passingPlays, which contains all data from pbp for passing plays only
passingPlays = pbp[pbp.playType == 'PASS']

# Extract rushingPlayerName from play description and drop into a new column of rushingPlays
passingPlays['passingPlayerName'] = passingPlays.description.str.extract('\d-(.+?)\s')
passingPlays['targetPlayerName1'] = passingPlays.description.str.extract('\w\w\s\d+-(.+?)\s')
passingPlays.targetPlayerName1 = passingPlays.targetPlayerName1.str.rstrip('.')
passingPlays['targetPlayerName2'] = passingPlays.description.str.extract('\w\w\s\d+-(.+?)\W$')
targetPlayerIsNull = passingPlays.targetPlayerName1.isnull()
passingPlays['targetPlayerIsNull'] = targetPlayerIsNull
passingPlays['targetPlayerName'] = (1 - passingPlays.targetPlayerIsNull) * passingPlays.targetPlayerName1 + \
                                   passingPlays.targetPlayerIsNull * passingPlays.targetPlayerName2
passingPlays = passingPlays.drop(['targetPlayerIsNull', 'targetPlayerName1', 'targetPlayerName2'], axis=1)


# Merge passingPlays and rushingPlays
allPlays = pd.concat([rushingPlays, passingPlays, otherPlays], ignore_index=False, sort=False)

# Drop 'challenger' (empty column) of allPlays
allPlays = allPlays.drop('challenger', axis=1)

# Create an isComplete column
allPlays['isComplete'] = allPlays.isPass - allPlays.isIncomplete

# Create binary column for isTouchdown (rows with TOUCHDOWN included in description)
touchdowns = allPlays.description.str.contains('TOUCHDOWN')
allPlays['isTouchdown'] = touchdowns.astype(int)

# Create binary column for isExtraPoint (rows with EXTRA POINT included in description)
extraPointAtt = allPlays.description.str.contains('EXTRA POINT')
allPlays['isExtraPointAtt'] = extraPointAtt.astype(int)

# Create binary column for isExtraPointSuccessful (rows with EXTRA POINT IS GOOD included in description)
extraPointSuccessful = allPlays.description.str.contains('EXTRA POINT IS GOOD')
allPlays['isExtraPointSuccessful'] = extraPointSuccessful.astype(int)

# Create binary column for isFieldGoalSuccessful (rows with FIELD GOAL IS GOOD included in description)
isFieldGoalSuccessful = allPlays.description.str.contains('FIELD GOAL IS GOOD')
allPlays['isFieldGoalSuccessful'] = isFieldGoalSuccessful.astype(int)

# Create binary column for isFieldGoalSuccessful (rows with FIELD GOAL IS GOOD included in description)
isSafety = allPlays.description.str.contains(', SAFETY')
allPlays['isSafety'] = isSafety.astype(int)

# Calculate and insert column into merged data frame (allPlays) for qtrSecRem, halfSecRem and gamSecRem
allPlays['qtrSecRem'] = allPlays.minute * 60 + allPlays.second
allPlays['gmSecRem'] = (4 - allPlays.quarter) * 900 + allPlays.qtrSecRem
timeMapQtr = {1: 1800, 2: 900, 3: 1800, 4: 900}
allPlays['halfSecRemBeg'] = allPlays.quarter.map(timeMapQtr)
allPlays['halfSecRem'] = allPlays.halfSecRemBeg - 900 + allPlays.qtrSecRem
allPlays = allPlays.drop('halfSecRemBeg', axis=1)

# Read .csv of 2019 schedule (sched) indexed by gameID into data frame (sched)
sched = pd.read_csv('/Users/samgreen/PycharmProjects/nfl1/2019_NFL_SCHEDULE.csv', index_col='gameID')

# Merge sched and allPlays based on gameID and drop Date column from sched
allPlays = pd.merge(allPlays, sched, on=['gameID']).drop(['Date'], axis=1)

# Sort allPlays for print and analysis
allPlays = allPlays.sort_values(by=['gameID', 'gameDate', 'quarter', 'minute', 'second', 'down'],
                                ascending=[True, True, True, False, False, True])

# Add column with binary indication of whether play was in red zone
allPlays['isGoalLineDown'] = allPlays.yardLine >= 90
allPlays.isGoalLineDown = allPlays.isGoalLineDown.astype(int)

# Add column with binary indication of whether play was in red zone
allPlays['isRedZone'] = allPlays.yardLine > 80
allPlays.isRedZone = allPlays.isRedZone.astype(int)

# Add binary column for whether halfSecRem <= 120 (2 min)
allPlays['isUTM'] = allPlays.halfSecRem <= 120
allPlays.isUTM = allPlays.isUTM.astype(int)

# Calculate scoring impact of each play in new column (scorePlay) in allPlays
allPlays['scorePlay'] = allPlays.isTouchdown * 6 + \
                        allPlays.isExtraPointSuccessful * 1 + \
                        allPlays.isTwoPointConversionSuccessful * 2 + \
                        allPlays.isSafety * 2 + \
                        allPlays.isFieldGoalSuccessful * 3

# Establish binary columns in allPlays for whether homeTeam or awayTeam on offense
homeTeamPoss = allPlays.offTeam == allPlays.homeTeam
allPlays['homeTeamPoss'] = homeTeamPoss
allPlays.homeTeamPoss = allPlays.homeTeamPoss.astype(int)
allPlays['awayTeamPoss'] = 1 - allPlays.homeTeamPoss

# Attribute scorePlay to one of two new columns: one for home team score and the other for away team score
allPlays['homeScorePlay'] = allPlays.homeTeamPoss * allPlays.scorePlay
allPlays['awayScorePlay'] = allPlays.awayTeamPoss * allPlays.scorePlay

# Calculate cumulative sum of home and away team scoring plays by gameID
allPlays['homeScoreCum'] = allPlays.groupby(['gameID'])['homeScorePlay'].cumsum()
allPlays['awayScoreCum'] = allPlays.groupby(['gameID'])['awayScorePlay'].cumsum()

# Calculate score differential for home team and away team by play
allPlays['homeScoreDiff'] = allPlays.homeScoreCum - allPlays.awayScoreCum
allPlays['awayScoreDiff'] = allPlays.homeScoreDiff * -1

# Calculate score differential for offensive team and defensive team
allPlays['offScoreDiff'] = allPlays.homeScoreDiff * allPlays.homeTeamPoss
allPlays['absScoreDiff'] = np.sqrt(allPlays.offScoreDiff ** 2).astype(int)

# Find index and associate scorePlay value of each non-zero instance in scorePlay
allPlays.scorePlay = allPlays.scorePlay.replace({0: np.nan})

# Establish binary column for whether a play is a scoring play
allPlays['isScore'] = allPlays.isTouchdown * 1 + \
                      allPlays.isExtraPointSuccessful * 1 + \
                      allPlays.isTwoPointConversionSuccessful * 1 + \
                      allPlays.isSafety * 1 + \
                      allPlays.isFieldGoalSuccessful * 1
allPlays['scoreTeam'] = allPlays.isScore * allPlays.offTeam
allPlays.isScore = allPlays.isScore.replace({0: np.nan})
games = allPlays.gameID.unique()
for g in allPlays.gameID:
    allPlays.nextScore = allPlays.scorePlay.bfill()
    allPlays.scoreTeam = allPlays.scoreTeam.bfill()

# Export data frames to a csv files
rushing_export_csv = rushingPlays.to_csv('rushing_export_dataframe.csv', index=True, header=True, index_label='playID')
passing_export_csv = passingPlays.to_csv('passing_export_dataframe.csv', index=True, header=True, index_label='playID')
all_export_csv = allPlays.to_csv('all_export_dataframe.csv', index=True, header=True, index_label='playID')

# Create new data frame (epPlaySet) equal to allPlays where quarter != 2 or 4
epPlaySet = allPlays[allPlays.quarter != 2]
epPlaySet = epPlaySet[epPlaySet.quarter != 4]

# Drop all rows from epPlaySet with playType of KICK OFF
epPlaySet = epPlaySet.drop(epPlaySet[epPlaySet.playType == 'KICK OFF'].index)

# Drop all rows from epPlaySet with absScoreDiff > 10
epPlaySet = epPlaySet.drop(epPlaySet[epPlaySet.absScoreDiff > 10].index)

# Drop all rows from epPlaySet with playType of KICK OFF
epPlaySet = epPlaySet.drop(epPlaySet[epPlaySet.isNoPlay == 1].index)

# Reorder columns in epPlaySet
epPlaySet = epPlaySet[['gameID',
                       'gameDate',
                       'quarter',
                       'minute',
                       'second',
                       'offTeam',
                       'defTeam',
                       'down',
                       'toGo',
                       'yardLine',
                       'seriesFirstDown',
                       'scorePlay',
                       'scoreTeam',
                       'nextScore',
                       'isScore',
                       'homeTeamPoss',
                       'awayTeamPoss',
                       'homeScorePlay',
                       'awayScorePlay',
                       'homeScoreCum',
                       'awayScoreCum',
                       'homeScoreDiff',
                       'awayScoreDiff',
                       'offScoreDiff',
                       'absScoreDiff',
                       'description',
                       'teamWin',
                       'seasonYear',
                       'yards',
                       'formation',
                       'playType',
                       'isRush',
                       'isPass',
                       'isIncomplete',
                       'passType',
                       'isSack',
                       'isChallenge',
                       'isChallengeReversed',
                       'isMeasurement',
                       'isInterception',
                       'isFumble',
                       'isPenalty',
                       'isTwoPointConversion',
                       'isTwoPointConversionSuccessful',
                       'rushDirection',
                       'yardLineFixed',
                       'yardLineDirection',
                       'isPenaltyAccepted',
                       'penaltyTeam',
                       'isNoPlay',
                       'penaltyType',
                       'penaltyYards',
                       'rushingPlayerName',
                       'passingPlayerName',
                       'targetPlayerName',
                       'isComplete',
                       'isTouchdown',
                       'isExtraPointAtt',
                       'isExtraPointSuccessful',
                       'isFieldGoalSuccessful',
                       'isSafety',
                       'qtrSecRem',
                       'gmSecRem',
                       'halfSecRem',
                       'week',
                       'time',
                       'awayTeam',
                       'homeTeam',
                       'winner',
                       'loser',
                       'isTie',
                       'ptsWinner',
                       'ptsLoser',
                       'ydWinner',
                       'turnoversWinner',
                       'ydLoser',
                       'turnoversLoser',
                       'isGoalLineDown',
                       'isRedZone',
                       'isUTM']]

# Print epPlaySet
print '\n'
print 'EP Play Set'
print epPlaySet
print epPlaySet.info()

# Export data frames to a csv files
ep_export_csv = epPlaySet.to_csv('ep_export_dataframe.csv', index=True, header=True, index_label='playID')

# Create new data frames for each down from epPlaySet
epPlaySet1down = epPlaySet[epPlaySet.down == 1]
epPlaySet2down = epPlaySet[epPlaySet.down == 2]
epPlaySet3down = epPlaySet[epPlaySet.down == 3]
epPlaySet4down = epPlaySet[epPlaySet.down == 4]

# Group each play set by yardLine and calculate average next score
ep1down = epPlaySet1down.groupby(['yardLine'])['nextScore'].mean()
ep2down = epPlaySet2down.groupby(['yardLine'])['nextScore'].mean()
ep3down = epPlaySet3down.groupby(['yardLine'])['nextScore'].mean()
ep4down = epPlaySet4down.groupby(['yardLine'])['nextScore'].mean()

# Produce scatter plots for EP by down by yardLine
plt.plot(ep1down, kind='scatter', title='Expected Points, 1st Down')
plt.show()
plt.plot(ep2down, kind='scatter', title='Expected Points, 2nd Down')
plt.show()
plt.plot(ep3down,kind='scatter', title='Expected Points, 3rd Down')
plt.show()
plt.plot(ep4down, type='scatter', title='Expected Points, 4th Down')
plt.show()


# -----------------------------------------------------------------
#  TO DOs:
#  --------
#   Add isSnap (or some way to calculate snap counts)
#   Expected points (EP) per play and winning probability (WP) by play
#   Targeted player - merge columns
#   Address error message:   SettingWithCopyWarning
#                               A value is trying to be set on a copy of a slice from a DataFrame.
#                               Try using .loc[row_indexer,col_indexer] = value instead
#   Set to 'NO PLAY' where appropriate
#   Need player database (esp player id)
#   Split passType into 2 columns: short/deep and left/right
#   Split rushDirection into 2 columns: left/right and tackle/guard
#   Calculate or pull intended air yards (total air yards regardless of whether pass attempt was completed)
#   Need to add timeouts remaining by team by play
#   Need log transformation of yards to go for 1st down
#   TEST play by play data output by game / learn to select last play of each game