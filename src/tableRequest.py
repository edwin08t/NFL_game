# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 21:39:22 2023

@author: Prime_Atom
"""

from sqlalchemy import create_engine
import pyodbc
import http.client
import pandas as pd
import json
import urllib
from datetime import datetime, timedelta
import numpy as np
import os
import ssl
import shutil



def avg_load(upper):
    
    import pandas as pd


    # Database connection parameters
    server = "Prime-Atom\ATOM"
    db = "NFL_TWO"
    user = "sa"
    password = "9053Ce713@"

    
    # Create the connection string for SQLAlchemy
    params = urllib.parse.quote_plus('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+server+';DATABASE='+db+';UID='+user+';PWD='+password)
    
    # Create SQLAlchemy engine
    engine = create_engine("mssql+pyodbc:///?odbc_connect=%s" % params)
    
    # Define the query
    weekRange = f"""
    select game_info_dim.id, season_year, week_title as week, 
    cast(concat(season_year,right('00' + cast(week_title as varchar(2)),2)) as int) as weekY, 
    concat(market, ' ', bet_name) as team_name, game_efficiency_fact.team_status, 
    game_touchdowns_fact.pass as td_pass, game_touchdowns_fact.rush as td_rush, 
    game_touchdowns_fact.total as td_total, points, cmp_pct, rating, 
    game_summary_fact.avg_gain as all_avg_gain, game_passing_fact.avg_yards as pass_avg, 
    game_passing_fact.net_yards as pass_net_yards, game_first_downs_fact.total as first_down_total, 
    goaltogo_attempts, goaltogo_successes, goaltogo_pct, game_efficiency_fact.redzone_attempts, 
    redzone_successes, redzone_pct, thirddown_pct, game_punts_fact.totals_attempts, point_spread, 
    case when touchdowns = 0 and totals_interceptions = 0 then cast(1 as float)
         when touchdowns = 0 and totals_interceptions > 0 then cast(0 as float)
         when touchdowns > 0 and totals_interceptions = 0 then cast(1 as float)
         when touchdowns > totals_interceptions then 1 - (totals_interceptions / cast(touchdowns as float))
         else cast(0 as float)
    end as QB_ratio, margin 
    from game_efficiency_fact
    left join game_teams_dim on game_efficiency_fact.id = game_teams_dim.id
    left join game_info_dim on game_efficiency_fact.game_id = game_info_dim.id
    join game_summary_fact on game_efficiency_fact.id = game_summary_fact.id
    join game_first_downs_fact on game_efficiency_fact.id = game_first_downs_fact.id
    join game_passing_fact on game_efficiency_fact.id = game_passing_fact.id
    join game_touchdowns_fact on game_efficiency_fact.id = game_touchdowns_fact.id
    join game_punts_fact on game_efficiency_fact.id = game_punts_fact.id
    join betting_splits on game_efficiency_fact.id = betting_splits.id
    join margin_table on game_efficiency_fact.id = margin_table.id
    where cast(concat(game_info_dim.season_year, right('00' + cast(game_info_dim.week_title as varchar(2)), 2)) as int) < {upper} 
    and cast(concat(game_info_dim.season_year, right('00' + cast(game_info_dim.week_title as varchar(2)), 2)) as int) > 200117
    order by game_info_dim.season_year desc, game_info_dim.week_title desc
    """
    
    # Execute the query and read data into a DataFrame
    main = pd.read_sql(weekRange, engine)
    
    # No need to commit or close the connection manually with SQLAlchemy
    # SQLAlchemy handles connection pooling and closing automatically
    
    return main

def avg_load_qb(upper):
    
    # Database connection parameters
    server = "Prime-Atom\ATOM"
    db = "NFL_TWO"
    user = "sa"
    password = "9053Ce713@"

   
    # Create the connection string for SQLAlchemy
    params = urllib.parse.quote_plus('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+server+';DATABASE='+db+';UID='+user+';PWD='+password)
   
    # Create SQLAlchemy engine
    engine = create_engine("mssql+pyodbc:///?odbc_connect=%s" % params)
    
    # Define the query
    weekRange = f"""
    select game_info_dim.id, season_year, week_title as week, 
    cast(concat(season_year,right('00' + cast(week_title as varchar(2)),2)) as int) as weekY, 
    concat(market, ' ', bet_name) as team_name, game_efficiency_fact.team_status, 
    game_touchdowns_fact.rush as td_rush, game_touchdowns_fact.total as td_total, points, 
    game_summary_fact.avg_gain as all_avg_gain, game_first_downs_fact.total as first_down_total, 
    goaltogo_attempts, goaltogo_successes, goaltogo_pct, game_efficiency_fact.redzone_attempts, 
    redzone_successes, redzone_pct, thirddown_pct, game_punts_fact.totals_attempts as punt_attempts, 
    point_spread, margin, attempts, completions, cmp_pct, totals_interceptions, sack_yards, rating, 
    touchdowns, game_passing_fact.avg_yards as qb_avg_yards, sacks, game_passing_fact.longest, 
    longest_touchdown, air_yards, yards, throw_aways, defended_passes, dropped_passes, spikes, 
    blitzes, hurries, knockdowns, pocket_time 
    from game_efficiency_fact 
    left join game_teams_dim on game_efficiency_fact.id = game_teams_dim.id 
    left join game_info_dim on game_efficiency_fact.game_id = game_info_dim.id 
    join game_summary_fact on game_efficiency_fact.id = game_summary_fact.id 
    join game_first_downs_fact on game_efficiency_fact.id = game_first_downs_fact.id 
    join game_passing_fact on game_efficiency_fact.id = game_passing_fact.id 
    join game_touchdowns_fact on game_efficiency_fact.id = game_touchdowns_fact.id 
    join game_punts_fact on game_efficiency_fact.id = game_punts_fact.id 
    join betting_splits on game_efficiency_fact.id = betting_splits.id 
    join margin_table on game_efficiency_fact.id = margin_table.id 
    where cast(concat(game_info_dim.season_year, right('00' + cast(game_info_dim.week_title as varchar(2)), 2)) as int) < {upper} 
    and cast(concat(game_info_dim.season_year, right('00' + cast(game_info_dim.week_title as varchar(2)), 2)) as int) > 200117 
    order by game_info_dim.season_year desc, game_info_dim.week_title desc
    """
    
    # Execute the query and read data into a DataFrame
    main = pd.read_sql(weekRange, engine)

    
    return main   


def all_load(upper):
    
    # Database connection parameters
    server = "Prime-Atom\ATOM"
    db = "NFL_TWO"
    user = "sa"
    password = "9053Ce713@"

   
    # Create the connection string for SQLAlchemy
    params = urllib.parse.quote_plus('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+server+';DATABASE='+db+';UID='+user+';PWD='+password)
   
    # Create SQLAlchemy engine
    engine = create_engine("mssql+pyodbc:///?odbc_connect=%s" % params)
    
    # Define the query
    weekRange = f"""
    select game_info_dim.id, game_info_dim.season_year, game_info_dim.week_title as week, 
    cast(concat(game_info_dim.season_year,right('00' + cast(game_info_dim.week_title as varchar(2)),2)) as int) as weekY, 
    concat(market, ' ', bet_name) as team_name, game_efficiency_fact.team_status, 
    game_touchdowns_fact.rush as td_rush, game_touchdowns_fact.total as td_total, points, 
    game_summary_fact.avg_gain as all_avg_gain, game_first_downs_fact.total as first_down_total, 
    goaltogo_attempts, goaltogo_successes, goaltogo_pct, game_efficiency_fact.redzone_attempts, 
    redzone_successes, redzone_pct, thirddown_pct, game_punts_fact.totals_attempts as punt_attempts, 
    point_spread, margin, attempts, completions, cmp_pct, totals_interceptions, sack_yards, rating, 
    touchdowns, game_passing_fact.avg_yards as qb_avg_yards, sacks, game_passing_fact.longest, 
    longest_touchdown, air_yards, yards, throw_aways, defended_passes, dropped_passes, spikes, 
    blitzes, hurries, knockdowns, pocket_time, Weighted_DVOA
    from game_efficiency_fact 
    left join game_teams_dim on game_efficiency_fact.id = game_teams_dim.id 
    left join game_info_dim on game_efficiency_fact.game_id = game_info_dim.id 
    join game_summary_fact on game_efficiency_fact.id = game_summary_fact.id 
    join game_first_downs_fact on game_efficiency_fact.id = game_first_downs_fact.id 
    join game_passing_fact on game_efficiency_fact.id = game_passing_fact.id 
    join game_touchdowns_fact on game_efficiency_fact.id = game_touchdowns_fact.id 
    join game_punts_fact on game_efficiency_fact.id = game_punts_fact.id 
    join betting_splits on game_efficiency_fact.id = betting_splits.id 
    join margin_table on game_efficiency_fact.id = margin_table.id 
	join DVOA_avg_fact on game_efficiency_fact.id = DVOA_avg_fact.id
    where cast(concat(game_info_dim.season_year, right('00' + cast(game_info_dim.week_title as varchar(2)), 2)) as int) < {upper}
    and cast(concat(game_info_dim.season_year, right('00' + cast(game_info_dim.week_title as varchar(2)), 2)) as int) > 200117 
    order by game_info_dim.season_year desc, game_info_dim.week_title desc
    """
    
    # Execute the query and read data into a DataFrame
    main = pd.read_sql(weekRange, engine)

    
    return main   
   
def bet_splits():
    
    # Database connection parameters
    server = "Prime-Atom\ATOM"
    db = "NFL_TWO"
    user = "sa"
    password = "9053Ce713@"

   
    # Create the connection string for SQLAlchemy
    params = urllib.parse.quote_plus('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+server+';DATABASE='+db+';UID='+user+';PWD='+password)
   
    # Create SQLAlchemy engine
    engine = create_engine("mssql+pyodbc:///?odbc_connect=%s" % params)
    
    betting_splits = """select game_id,betting_splits.status, point_spread  from betting_splits
    """
    
    #betting_splits with all columns 
    betting_splits = pd.read_sql(betting_splits, engine)
    
    
    return betting_splits


def schedule(week,year):
    #path to the schedule file
    path =  'Z:/NFL Project/NFL_Two/Seasonal_Updates/yearlySchedule/{}.json'.format(year)
    
    week = int(week)
    
    #go to json schedule file
       
    data = json.load(open(path))
    #------------------game_table----------------#
    game_table =pd.json_normalize(data, record_path=['weeks','games'],meta=[['weeks','title']])
    if 'scoring.periods' in game_table.columns:
        # Proceed with your operation
        game_table.drop('scoring.periods', axis=1, inplace=True)
    game_table = game_table.rename(columns={'id':"game_id"})
       
       ##format date to time
    game_table['scheduled'] = pd.to_datetime(game_table['scheduled'])
       
       #create a new schedule dataframe
       
    game_table['scheduled'] = game_table['scheduled'].dt.tz_localize(None)
    game_table['scheduled'] = game_table['scheduled'] - timedelta(hours=7, minutes=0)
    game_table['start_time'] = pd.to_datetime(game_table['scheduled'],format= '%H:%M:%S' ).dt.time
    game_table['game_date'] = game_table['scheduled'].dt.date
  
    
    
    game_table = game_table.rename(columns={'weeks.title':"week_title"})
    
    game_table  = game_table.loc[game_table ['status'] != 'cancelled']
    
    
    
    schedule_table = game_table[['game_id','game_date','start_time','home.id','home.name','away.id','away.name','week_title']]
     
       
       
    #rename column home.name, away.name
    schedule_table.columns=[it.replace('home.name','home_team') for it in schedule_table.columns]
    schedule_table.columns=[it.replace('away.name','away_team') for it in schedule_table.columns]
    schedule_table.columns=[it.replace('home.id','home_id') for it in schedule_table.columns]
    schedule_table.columns=[it.replace('away.id','away_id') for it in schedule_table.columns]
       
    #add week column to schedule table
    schedule_table.loc[:, 'week_title'] = schedule_table['week_title'].astype(str).astype(int)
    
    #filter future schedule table for the current week####################################
    schedule_table  = schedule_table.loc[schedule_table ['week_title'] == week]
    #######################################################################################
    
    return schedule_table

def team_name(year):
    # Database connection parameters
    server = "Prime-Atom\ATOM"
    db = "NFL_TWO"
    user = "sa"
    password = "9053Ce713@"

   
    # Create the connection string for SQLAlchemy
    params = urllib.parse.quote_plus('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+server+';DATABASE='+db+';UID='+user+';PWD='+password)
   
    # Create SQLAlchemy engine
    engine = create_engine("mssql+pyodbc:///?odbc_connect=%s" % params)

    
        
        
    test = """SELECT team_id,CONCAT(market,' ',team_name) as team
    FROM game_teams_dim
    
    join game_info_dim
        on game_teams_dim.game_id = game_info_dim.id

    where season_year = {}
    Group by team_id,CONCAT(market,' ',team_name)
    ORDER BY CONCAT(market,' ',team_name) ASC""".format(year)
                
       
    main = pd.read_sql(test, engine)
    
    
    return main

def qb_request(week,year):

    
    intWeek  = int(week)
    
    data = json.load(open('Z:/NFL Project/NFL_Two/Seasonal_Updates/weeklyDepth/{}_{}.json'.format(year,intWeek)))
    mainOne = pd.DataFrame()


    # Normalize the JSON data into a DataFrame
    weeklydepth = pd.json_normalize(
        data,
        record_path=['teams', 'offense', ['position', 'players']],
        meta=[['teams', 'id'], ['teams', 'name']],
        errors='ignore'
    )

    # Filter the DataFrame to include only quarterbacks (QB)
    weeklydepth = weeklydepth[weeklydepth['position'] == 'QB']

    # Extract and deduplicate team names
    activeTeam = weeklydepth[['teams.name']].drop_duplicates()

    # Separate starting and backup quarterbacks
    qbOne = weeklydepth[weeklydepth['depth'] == 1]
    backUp = weeklydepth[weeklydepth['depth'] == 2]

    # Determine which teams do not have a starting QB and need backup QBs
    teams_without_qbOne = activeTeam[~activeTeam['teams.name'].isin(qbOne['teams.name'])]
    backUp_needed = backUp[backUp['teams.name'].isin(teams_without_qbOne['teams.name'])]

    # Combine starting QBs with the necessary backup QBs
    qbOne = pd.concat([qbOne, backUp_needed], ignore_index=True)

    # Remove duplicates based on team name, keeping the first occurrence
    weeklydepth = qbOne.drop_duplicates(subset=['teams.name'], keep='first')
     
     
    # Database connection parameters
    server = "Prime-Atom\ATOM"
    db = "NFL_TWO"
    user = "sa"
    password = "9053Ce713@"

   
    # Create the connection string for SQLAlchemy
    params = urllib.parse.quote_plus('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+server+';DATABASE='+db+';UID='+user+';PWD='+password)
   
    # Create SQLAlchemy engine
    engine = create_engine("mssql+pyodbc:///?odbc_connect=%s" % params)

    
     
    for index, row in weeklydepth.iterrows():
        
        
        test = """SELECT top (9) game_info_dim.id ,team_id,season_year,week_title,cast(concat(season_year,right('00' +cast(week_title as varchar(2)),2))as int) as weekY,
        player_name,player_id,rating,cmp_pct,completions,attempts,yards,avg_yards as qb_avg_yards,air_yards,touchdowns,longest_touchdown,longest,interceptions as totals_interceptions ,
        sacks,sack_yards,blitzes,hurries,pocket_time,defended_passes,knockdowns,dropped_passes,throw_aways,spikes
        FROM NFL_TWO.dbo.player_passing_fact
        
        join game_info_dim
             on player_passing_fact.game_id = game_info_dim.id 
        
        where player_id = '{}' and cast(concat(season_year,right('00' +cast(week_title as varchar(2)),2))as int) < {}{} 
        order by season_year desc,week_title desc""".format(row['id'],year,week)
        
        
        
        new_row = pd.read_sql(test, engine)
        
        mainOne = pd.concat([mainOne, new_row])
                
     
    weeklydepth = weeklydepth.rename(columns={'id':"player_id"})
    weeklydepth = weeklydepth.rename(columns={'teams.id':"team_id"})
    
  
    mainOne["air_yards"] = mainOne["air_yards"].astype(np.int64)
    mainOne["completions"] = mainOne["completions"].astype(np.int64)
    mainOne["attempts"] = mainOne["attempts"].astype(np.int64)
    mainOne["yards"] = mainOne["yards"].astype(np.int64)
    mainOne["touchdowns"] = mainOne["touchdowns"].astype(np.int64)
    mainOne["longest_touchdown"] = mainOne["longest_touchdown"].astype(np.int64)
    mainOne["longest"] = mainOne["longest"].astype(np.int64)
    mainOne["sack_yards"] = mainOne["sack_yards"].astype(np.int64)
    mainOne["sacks"] = mainOne["sacks"].astype(np.int64)
    mainOne["blitzes"] = mainOne["blitzes"].astype(np.int64)
    mainOne["hurries"] = mainOne["hurries"].astype(np.int64)
    mainOne["defended_passes"] = mainOne["defended_passes"].astype(np.int64)
    mainOne["knockdowns"] = mainOne["knockdowns"].astype(np.int64)
    mainOne["dropped_passes"] = mainOne["dropped_passes"].astype(np.int64)
    mainOne["totals_interceptions"] = mainOne["totals_interceptions"].astype(np.int64)
    mainOne["throw_aways"] = mainOne["throw_aways"].astype(np.int64)
    mainOne["spikes"] = mainOne["spikes"].astype(np.int64)
    
    
    mainOne.drop(['id','season_year','week_title','weekY','player_name','team_id'], axis=1, inplace=True)
    
    mainOne = mainOne.groupby('player_id', as_index=False).mean()
    
    
    weeklydepth = pd.merge(weeklydepth,mainOne,how='left', on=['player_id'])
     
    
    
          
    ssl._create_default_https_context = ssl._create_unverified_context
    
    
    #path to the schedule file
    path =  'Z:/NFL Project/NFL_Two/Seasonal_Updates/yearlySchedule/{}.json'.format(year)


    data = json.load(open(path))
    #------------------game_table----------------#
    game_table =pd.json_normalize(data, record_path=['weeks','games'],meta=[['weeks','title']])
    
    if 'scoring.periods' in game_table.columns:
        # Proceed with your operation
        game_table.drop('scoring.periods', axis=1, inplace=True)
    game_table = game_table.rename(columns={'id':"game_id"})

    ##format date to time
    game_table['scheduled'] = pd.to_datetime(game_table['scheduled'])

    #create a new schedule dataframe

    game_table['scheduled'] = game_table['scheduled'].dt.tz_localize(None)
    game_table['scheduled'] = game_table['scheduled'] - timedelta(hours=7, minutes=0)
    game_table['game_date'] = game_table['scheduled'].dt.date

    
    game_table = game_table[['game_id','game_date','home.name','away.name','weeks.title','home.id','away.id']]
    #rename column home.name, away.name
    game_table.columns=[it.replace('home.name','home_team') for it in game_table.columns]
    game_table.columns=[it.replace('away.name','away_team') for it in game_table.columns]
    game_table.columns=[it.replace('home.id','home_id') for it in game_table.columns]
    game_table.columns=[it.replace('away.id','away_id') for it in game_table.columns]
    game_table.columns=[it.replace('weeks.title','weeks_title') for it in game_table.columns]
    game_table['weeks_title'] = game_table['weeks_title'].astype(str).astype(int)

    game_table = game_table.loc[game_table['weeks_title'] == intWeek]
    
    
    home = game_table[['game_id','home_id','game_date','home_team']].copy()

    home.loc[:, 'Status'] = 'Home'

    home = home.rename(columns={'home_id':"team_id",'home_team':"team_team"})

    away = game_table[['game_id','away_id','game_date','away_team']].copy()

    away.loc[:, 'Status'] = 'Away'

    


    away = away.rename(columns={'away_id':"team_id",'away_team':"team_team"})

    game_table = pd.concat([home, away], ignore_index=True)
    



    game_table['id'] =  game_table['game_id'] +game_table['team_id']

    game_table = game_table[['team_id','team_team']]
    
 
    game_table = game_table.rename(columns={'team_team':"team"})
    
    
    weeklydepth = pd.merge(game_table,weeklydepth, how='left', on=['team_id'])
    
    
     
    weeklydepth.drop(['player_id','name','jersey','position','sr_id','depth','team_id','teams.name'], axis=1, inplace=True)
     
    first_column = weeklydepth.pop('team') 
     
    weeklydepth.insert(0, 'team', first_column) 

    
    weeklydepth = weeklydepth.rename(columns={'team':"team_name"})
    
    return weeklydepth

  


def DVOA_weekly(week,year):
    
    week = int(week)
    

    #path to the schedule file
    path =  'Z:/NFL Project/NFL_Two/Seasonal_Updates/yearlySchedule/{}.json'.format(year)


    #go to json schedule file
       
    data = json.load(open(path))
            
    #------------------game_table----------------#
    game_table = pd.json_normalize(data, record_path=['weeks', 'games'], meta=[['weeks', 'title']])
    if 'scoring.periods' in game_table.columns:
        # Proceed with your operation
        game_table.drop('scoring.periods', axis=1, inplace=True)
    game_table = game_table.rename(columns={'id': "game_id"})
    
   

    # Format date to time
    game_table['scheduled'] = pd.to_datetime(game_table['scheduled'])
    game_table['scheduled'] = game_table['scheduled'].dt.tz_localize(None)
    game_table['scheduled'] = game_table['scheduled'] - timedelta(hours=7)
    game_table['game_date'] = game_table['scheduled'].dt.date

    game_table = game_table[['game_id', 'game_date', 'home.name','home.alias', 'away.name','away.alias', 'weeks.title', 'home.id', 'away.id']]

    # Rename columns
    game_table.columns = [col.replace('home.name', 'home_team').replace('away.name', 'away_team')
                            .replace('home.id', 'home_id').replace('away.id', 'away_id')
                            .replace('home.alias', 'home_alias').replace('away.alias', 'away_alias')
                            .replace('weeks.title', 'weeks_title') for col in game_table.columns]

    game_table['weeks_title'] = game_table['weeks_title'].astype(int)
    game_table = game_table.loc[game_table['weeks_title'] == week]

    home = game_table[['game_id', 'home_id', 'game_date', 'home_team','home_alias']].copy()
    away = game_table[['game_id', 'away_id', 'game_date', 'away_team', 'away_alias']].copy()

    home.loc[:, 'Status'] = 'Home'
    home = home.rename(columns={'home_id': "team_id", 'home_team': "team_team",'home_alias': 'alias'})

    away.loc[:, 'Status'] = 'Away'
    away = away.rename(columns={'away_id': "team_id", 'away_team': "team_team", 'away_alias': 'alias'})

    game_table = pd.concat([home, away], ignore_index=True)

    
    game_table = game_table.rename(columns={'alias': "Team", 'team_team': "team_name"})
    
    game_table = game_table[['Team','team_name']]
    
    
    

    if week <= 5:
        dvoa = pd.read_csv('Z:/NFL Project/NFL_Two/Completed/Dave/{} Team DAVE Ratings, Overall after Week {}.csv'.format(year,week))
        
        dvoa = dvoa.rename(columns={'TEAM':"Team",'TOT DAVE':"Weighted_DVOA"})
        
        dvoa = dvoa[['Team','Weighted_DVOA']]
        
    else:
        pastWeek = week - 1
        
        dvoa = pd.read_csv('Z:/NFL Project/NFL_Two/Completed/DVOA/{} Team DVOA Ratings, Overall after Week {}.csv'.format(year,pastWeek))
       
        dvoa = dvoa.rename(columns={'TEAM':"Team",'WEIGHTED DVOA':"Weighted_DVOA"}) 

    
        dvoa['Weighted_DVOA'] = dvoa['Weighted_DVOA'].str.rstrip('%').astype('float') / 100.0
        
        dvoa = dvoa[['Team','Weighted_DVOA']]
        
    
    
    
    dvoa.loc[dvoa['Team'] == 'JAX', 'Team'] = 'JAC'
    dvoa.loc[dvoa['Team'] == 'LAR', 'Team'] = 'LA'
    
    dvoa = pd.merge(game_table,dvoa, on ='Team',how='left')   
    
    game_table.to_csv(r'Z:/NFL Project/NFL_Two/test.csv')
    
    dvoa = dvoa[['team_name','Weighted_DVOA']]
    
    
    return dvoa

     

    
def upperRange(week,year):
     #change year to str since comes as an object
     year = str(year)
     
     
     #create uppper limit YYYYMM
     maxCurrent = year + week
     
     maxCurrent  = int(maxCurrent)
     
     
     return maxCurrent
 

def minNine(week,year):   
    
     #change year to int
     year = int(year)
     
     #change week to int
     week = int(week)
     
     #create a 10 day gap
     minCurrent = -5
     
     #if statement for weeks that are in prior year
     if (minCurrent + week) < 0:   #if negative number than its the prior year
         
         #current year minus one
         year = year - 1
         
         #lower limit plus 18 weeeks
         minCurrent = (minCurrent + week)+18
     
     else:
         
         #just add the current and lower limit
         minCurrent = ( week+ minCurrent )
     
     #change year back to string
     year = str(year)
     
     #change the week to string
     minCurrent = str(minCurrent)
     
     #make week is to char if not fill with 0
     minCurrent = minCurrent.zfill(2)
     
     #combine year and week to form the lower limit
     minCurrent = year+minCurrent
     
     #turn lower limit to int
     minCurrent  = int(minCurrent)
     
     return minCurrent
 
     
def check_main(current,year):
    
    current = int(current)
    
    current = current -1
    
    if current != 0:
        
        # Database connection parameters
        server = "Prime-Atom\ATOM"
        db = "NFL_TWO"
        user = "sa"
        password = "9053Ce713@"
    
        
        # Create the connection string for SQLAlchemy
        params = urllib.parse.quote_plus('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+server+';DATABASE='+db+';UID='+user+';PWD='+password)
        
        # Create SQLAlchemy engine
        engine = create_engine("mssql+pyodbc:///?odbc_connect=%s" % params)
        
        main = """select prediction_first_fact.game_id,prediction_first_fact.game_date,prediction_first_fact.home_team,prediction_first_fact.away_team,
        prediction_first_fact.week_title, prediction_first_fact.margin, prediction_first_fact.total,
        betting_splits.point_spread,betting_splits.Totals from prediction_first_fact
        join betting_splits
        on betting_splits.game_id = prediction_first_fact.game_id
        
        
        where week_title = {} and Status = 'Home' and season_year = {} """.format(current,year)
        
        points = """select *  from game_summary_fact
        
        """
        
        
        
        #main with all columns 
        main = pd.read_sql(main, engine)
        
     
        points  = pd.read_sql(points , engine)
        
        
        #creating home table with just points
        HomePoints = points[['game_id','points','team_status']]
        HomePoints = HomePoints.loc[points['team_status'] == 'Home']
        HomePoints = HomePoints.rename(columns={'home_id':"id",'points':"home_points"})
        HomePoints.drop(['team_status'], axis=1, inplace=True)
        
        #creating away table with just points
        AwayPoints = points[['game_id','points','team_status']]
        AwayPoints = AwayPoints.loc[points['team_status'] == 'Away']
        AwayPoints = AwayPoints.rename(columns={'away_id':"id",'points':"away_points"})
        AwayPoints.drop(['team_status'], axis=1, inplace=True)
        
        
        
        #merge home and away data to get home predictions
        pointTable = pd.merge(HomePoints,AwayPoints,how='left', on=['game_id'])
        
        pointTable = pd.merge(main,pointTable,how='left', on=['game_id'])
        
        
        
        pointTable['game_margin'] = pointTable['away_points'] - pointTable['home_points']
        
        pointTable['game_total'] = pointTable['away_points'] + pointTable['home_points']
        
        
        
        pointTable = pointTable.rename(columns={'margin':"pred_margin",'Totals':"bet_totals",'total':"pred_total"})
        
        
        pointTable = pointTable[['game_id','game_date','home_team','away_team','week_title','point_spread','pred_margin','game_margin',
                                 'bet_totals','pred_total','game_total']]
        
        pointTable['pred_margin'] = pointTable['pred_margin'].round(0).astype(int)
        pointTable['pred_total'] = pointTable['pred_total'].round(0).astype(int)
        
        
        
         
        def flag_df(pointTable):
            
            #if the game margin, spread, predicted score all negative 
            if pointTable['point_spread'] < 0 and pointTable['pred_margin'] < 0 and pointTable['game_margin'] < 0:
                      
                      # if all three match than return 1
                      if pointTable['game_margin'] == pointTable['point_spread'] == pointTable['pred_margin']  :
                          return 1
                      
                      # if the game margin is larger than the spread. check if the predicted
                      # socre is larger than the spread and return a 1
                      elif pointTable['game_margin'] > pointTable['point_spread']:
                          if pointTable['pred_margin'] > pointTable['point_spread']:
                              return 1
                          else:
                              return 0
                          
                      #check if the the game margin is closer to the predicted score vs the spread
                      elif pointTable['game_margin']*-1 > pointTable['point_spread']*-1:
                          if pointTable['pred_margin']*-1 > pointTable['point_spread']*-1:
                              return 1
                          else:
                             return 0
                      else:
                         return 0
                     
            #if the game margin, spread, predicted score all postive
            elif pointTable['point_spread'] > 0 and pointTable['pred_margin'] > 0 and pointTable['game_margin'] > 0:
                      # if all three match than return 1
                      if pointTable['game_margin'] == pointTable['point_spread'] == pointTable['pred_margin']:
                          return 1
                      
                      #check if the the game margin is closer to the predicted score vs the spread
                      elif pointTable['game_margin'] > pointTable['point_spread']:
                          if pointTable['pred_margin'] > pointTable['point_spread']:
                              return 1
                          else:
                             return 0
                      else:
                         return 0
               
            #when the spread is negative and the predicted soores is postive as well as the game socre 
            elif pointTable['point_spread'] < 0 and pointTable['pred_margin'] > 0 and pointTable['game_margin'] > 0:
                      return 1
            
            #when the spread is postive and the predicted soores is negative as well as the game socre 
            elif pointTable['point_spread'] > 0 and pointTable['pred_margin'] < 0 and pointTable['game_margin'] < 0:
                      return 1
            #when the spread and predicted score are negative and game score is postive
            #check to see if the predicted score is larger than the spread
            elif pointTable['point_spread'] < 0 and pointTable['pred_margin'] < 0 and pointTable['game_margin'] > 0:
                      if pointTable['point_spread'] < pointTable['pred_margin'] :
                          return 1
                      else:
                          return 0
            
            #when the home team is suppose to when by a ceratin margin
            # but they win with less or lose.
            elif pointTable['point_spread'] < 0 and pointTable['pred_margin'] > 0 and pointTable['game_margin'] < 0:
                      if pointTable['point_spread'] < pointTable['game_margin'] :
                          return 1
                      else:
                          return 0
                      
            else:
                return 0
                
        
            
        def flag_total(pointTable):
            betGame  = pointTable['game_total'] -  pointTable['bet_totals'] 
            predGame =  pointTable['pred_total'] -  pointTable['bet_totals'] 
            
            if betGame == 0:
                return 0
            elif betGame < 0 and predGame < 0:
                    return 1
            elif betGame > 0 and predGame > 0: 
                 return 1
            
            else:
                return 0
            
        def home_ATS(pointTable):
            
            if pointTable['point_spread'] < 0:
                if pointTable['point_spread'] > pointTable['game_margin']:
                    return 1
                else:
                    return 0
            elif pointTable['point_spread'] > pointTable['game_margin']:
                return 1
            else:
                return 0
            
        def over(pointTable):
            
            if pointTable['bet_totals'] < pointTable['game_total']:
                return 1
            else:
                return 0
        
          
            
        pointTable['margin_won'] = pointTable.apply(flag_df, axis = 1)
        
        pointTable['total_won'] = pointTable.apply(flag_total, axis = 1)
        
        pointTable['home_ATS'] = pointTable.apply(home_ATS, axis = 1)
        
        pointTable['home_over'] = pointTable.apply(over, axis = 1)
        
        pointTable['away_over'] = pointTable.apply(over, axis = 1)
        
        pointTable.loc[pointTable['home_ATS'] == 1, 'away_ATS'] = 0  
        
        pointTable.loc[pointTable['home_ATS'] == 0, 'away_ATS'] = 1
        
        pointTable['home_ATS_over'] = pointTable.apply(over, axis = 1)
    
        pointTable['away_ATS_over'] = pointTable.apply(over, axis = 1)
        
        pointTable['season_year'] = year
    
        col = pointTable.pop("season_year")
    
        pointTable.insert(4, col.name, col)
        
     
        
       
        server = "Prime-Atom\ATOM"
        db = "NFL_TWO"
        user = "sa"
        password = "9053Ce713@"
        port = "1433"
    
        conn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER=' + server + ';PORT=' + port + ';DATABASE=' + db +';UID=' + user + ';PWD=' + password)
        cursor = conn.cursor()
        
        
        #--------------------------------------------------------------------Summary------------------------------------------------------------------------#
        # Insert Dataframe into SQL Server:
            
        ## check if file is not blank
        
        ##loop through file and submit to database
        for index, row in pointTable.iterrows():
            
             cursor.execute("""INSERT INTO NFL_TWO.dbo.valid_first_fact(game_id,game_date,home_team,away_team,season_year,week_title,point_spread,pred_margin,
                            game_margin,bet_totals,pred_total,game_total,margin_won,total_won,home_ATS,away_ATS,home_ATS_over,away_ATS_over)
                            values(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                            row.game_id,row.game_date,row.home_team,row.away_team,row.season_year,row.week_title,row.point_spread,row.pred_margin,row.game_margin,
                            row.bet_totals,row.pred_total,row.game_total,row.margin_won,row.total_won,row.home_ATS,row.away_ATS,row.home_ATS_over,row.away_ATS_over)
        
        
        ## submit line, close cursor and disconnect connection            
        conn.commit()
        cursor.close()
        conn.close()  
        
def move_file():
    ### Path to JSON files
    rootdir = 'Z:/NFL Project/NFL_Two/Current/Games/'
    
    
    ##start looping through Json files
    for filename in os.listdir(rootdir): 
        f = os.path.join(rootdir, filename)
        if os.path.isfile(f):
            shutil.move('Z:/NFL Project/NFL_Two/Current/Games/{}'.format(filename), 'Z:/NFL Project/NFL_Two/Completed/Games/{}'.format(filename))


    