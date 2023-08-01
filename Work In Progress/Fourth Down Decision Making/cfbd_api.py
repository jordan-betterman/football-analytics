import pandas as pd
import numpy as np
import requests


dataframe = pd.DataFrame()

big_ten_teams = ["Minnesota", "Wisconsin", "Ohio State", "Penn State", "Michigan", "Michigan State", "Nebraska", "Northwestern", "Illinois", "Indiana", "Iowa", "Maryland", "Purdue", "Rutgers"]

for team in big_ten_teams:
    print(team)
    for year in range(2017,2023):
        #print(year)
        for i in range(1,14):
            plays = requests.get(url=f'https://api.collegefootballdata.com/plays?seasonType=regular&week={i}&year={year}&team={team}', 
                                        headers={'Authorization' : 'Bearer ti+f5GzjSdgTT5Lc1m5PcOE4kcRZrs327v2VUYK8sUo2sSIEvnZZQ08FUsINUCYL'}).json()

            for play in plays: 
                play["minutes"] = play['clock']["minutes"]
                play["seconds"] = play['clock']['seconds']

            frame = pd.DataFrame(plays) 
            dataframe = pd.concat([dataframe, frame])
    
dataframe.to_csv('big_ten_pbp.csv')

dataframe.head()