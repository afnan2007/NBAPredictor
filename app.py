import streamlit as st
import pandas as pd
import sklearn
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import TimeSeriesSplit

st.set_page_config(page_title="NBA Matchups Predictor With ML", layout="wide")

# Load and preprocess the data
data = pd.read_csv("nba_games.csv", index_col=0)
data = data.sort_values("date")
data = data.reset_index(drop=True)
del data["mp.1"]
del data["mp.opp.1"]
del data["index_opp"]

def add_target(team):
    team["target"] = team["won"].shift(-1)
    return team  

data = data.groupby("team", group_keys=False).apply(add_target)
data["target"][pd.isnull(data["target"])] = 2
data["target"] = data["target"].astype(int, errors="ignore")

nulls = pd.isnull(data)
nulls = nulls.sum()
nulls = nulls[nulls > 0]
valid_columns = data.columns[~data.columns.isin(nulls.index)]
data = data[valid_columns].copy()

# Feature selection setup
ridge = RidgeClassifier(alpha=1)
split = TimeSeriesSplit(n_splits=3)
sfs = SequentialFeatureSelector(ridge, n_features_to_select=30, direction="forward", cv=split)

removed_columns = ["season", "date", "won", "target", "team", "team_opp"]
selected_columns = data.columns[~data.columns.isin(removed_columns)]

scaler = MinMaxScaler()
data[selected_columns] = scaler.fit_transform(data[selected_columns])
sfs.fit(data[selected_columns], data["target"])
predictors = list(selected_columns[sfs.get_support()])

# Compute rolling averages
data_rolling = data[list(selected_columns) + ["won", "team", "season"]]

def team_averages(team):
    rolling = team.rolling(10).mean()
    return rolling

data_rolling = data_rolling.groupby(["team", "season"], group_keys=False).apply(team_averages)
rolling_cols = [f"{col}_10" for col in data_rolling.columns]
data_rolling.columns = rolling_cols

data = pd.concat([data, data_rolling], axis=1)
data = data.dropna()

def shift_col(team, col_name):
    next_col = team[col_name].shift(-1)
    return next_col

def add_col(data, col_name):
    return data.groupby("team", group_keys=False).apply(lambda x: shift_col(x, col_name))

data["home_next"] = add_col(data, "home")
data["team_opp_next"] = add_col(data, "team_opp")
data["date_next"] = add_col(data, "date")

data = data.copy()

full = data.merge(data[rolling_cols + ["team_opp_next", "date_next", "team"]], 
                  left_on=["team", "date_next"], 
                  right_on=["team_opp_next", "date_next"])

removed_columns = list(full.columns[full.columns.dtypes == "object"]) + removed_columns
selected_columns = full.columns[~full.columns.isin(removed_columns)]
sfs.fit(full[selected_columns], full["target"])

predictors = list(selected_columns[sfs.get_support()])

# Streamlit Interface
st.title("NBA Matchups Predictor")
team_list = sorted(data["team"].unique())
team1 = st.selectbox("Select Team 1", team_list)
team2 = st.selectbox("Select Team 2", team_list)
game_date = st.date_input("Select Game Date")

# Predict outcome based on user input
if st.button("Predict Outcome"):
    date_str = game_date.strftime('%Y-%m-%d')
    match_data = full[(full["team"] == team1) & (full["team_opp_next"] == team2) & (full["date_next"] == date_str)]
    
    if match_data.empty:
        st.write("No data available for this matchup and date.")
    else:
        prediction = ridge.predict(match_data[predictors])[0]
        if prediction == 1:
            st.write(f"{team1} won the game against {team2} on {date_str}.")
        else:
            st.write(f"{team2} won the game against {team1} on {date_str}.")
