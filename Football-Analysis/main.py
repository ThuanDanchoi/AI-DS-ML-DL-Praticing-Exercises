import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

players_data = pd.read_csv('data/players.csv')
appearances_data = pd.read_csv('data/appearances.csv')
games_data = pd.read_csv('data/games.csv')

full_players = players_data.merge(appearances_data, on='player_id')
full_players = full_players.merge(games_data[['game_id', 'season']], on='game_id')

full_players = full_players.sort_values(by='market_value_in_eur', ascending=False)
full_players = full_players.fillna().sum()

print(full_players.isna().sum())

