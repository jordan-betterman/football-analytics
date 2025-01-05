import os
from dotenv import load_dotenv
import pandas as pd
import requests

load_dotenv()

big_ten_teams = [
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


def play_by_play(destination_directory="big_ten_pbp.csv", teams=big_ten_teams):
    dataframe = pd.DataFrame()

    for team in teams:
        print(team)
        for year in range(2017, 2023):
            for i in range(1, 14):
                plays = requests.get(
                    url=f"https://api.collegefootballdata.com/plays?seasonType=regular&week={i}&year={year}&team={team}",
                    headers={"Authorization": f"Bearer {os.getenv("CFBD_API_KEY")}"},
                ).json()

                for play in plays:
                    play["minutes"] = play["clock"]["minutes"]
                    play["seconds"] = play["clock"]["seconds"]

                frame = pd.DataFrame(plays)
                dataframe = pd.concat([dataframe, frame])

    dataframe.to_csv(destination_directory)

    return dataframe


def get_player_season_stats():
    dataframe = pd.DataFrame()
    for year in range(2021, 2024):
        player_stats = requests.get(
            url=f"https://api.collegefootballdata.com/stats/player/season?year={year}",
            headers={"Authorization": f"Bearer {os.getenv("CFBD_API_KEY")}"},
        ).json()
        player_dataframe = pd.DataFrame(player_stats)
        player_dataframe["season"] = year
        dataframe = pd.concat([dataframe, player_dataframe])
    dataframe.to_csv("work_in_progress/player_stats.csv", index=False)


def get_transfer_portal_players():
    dataframe = pd.DataFrame()
    for year in range(2021, 2024):
        portal_players = requests.get(
            url=f"https://api.collegefootballdata.com/player/portal?year={year}",
            headers={"Authorization": f"Bearer {os.getenv("CFBD_API_KEY")}"},
        ).json()
        portal_dataframe = pd.DataFrame(portal_players)
        dataframe = pd.concat([dataframe, portal_dataframe])
    dataframe.to_csv("work_in_progress/portal_players.csv", index=False)


def get_players_usage():
    dataframe = pd.DataFrame()
    for year in range(2021, 2024):
        usage_players = requests.get(
            url=f"https://api.collegefootballdata.com/player/usage?year={year}",
            headers={"Authorization": f"Bearer {os.getenv("CFBD_API_KEY")}"},
        ).json()
        usage_dataframe = pd.DataFrame(usage_players)
        dataframe = pd.concat([dataframe, usage_dataframe])
    dataframe.to_csv("work_in_progress/usage_players.csv", index=False)
