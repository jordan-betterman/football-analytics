import numpy as np
import pandas as pd
import requests

BIG_TEN_TEAMS = [
    "Minnesota",
    "Wisconsin",
    "Ohio State",
    "Penn State",
    "Michigan",
    "Michigan State",
    "Nebraska",
    "Northwestern",
    "Illinois",
    "Indiana",
    "Iowa",
    "Maryland",
    "Purdue",
    "Rutgers",
]


def play_by_play(teams=BIG_TEN_TEAMS):
    dataframe = pd.DataFrame()

    for team in teams:
        print(team)
        for i in range(1, 6):
            plays = requests.get(
                url=f"https://api.collegefootballdata.com/plays?seasonType=regular&week={i}&year=2023&team={team}",
                headers={
                    "Authorization": "Bearer ti+f5GzjSdgTT5Lc1m5PcOE4kcRZrs327v2VUYK8sUo2sSIEvnZZQ08FUsINUCYL"
                },
            ).json()

            for play in plays:
                play["minutes"] = play["clock"]["minutes"]
                play["seconds"] = play["clock"]["seconds"]

            frame = pd.DataFrame(plays)
            dataframe = pd.concat([dataframe, frame])

    return dataframe
