import streamlit as st
import pandas as pd
import datetime 
import sklearn
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import TimeSeriesSplit

st.set_page_config(page_title="NBA Matchup Predictor", layout="wide")

gradient = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(45deg, red, blue);
}

[data-testid="stHeader"] {
    background-color: rgba(0,0,0,0)
}
</style>
"""

title = """
<style>
.title {
    text-align: center;
}
</style>
"""

st.markdown(gradient, unsafe_allow_html=True)
st.markdown(title, unsafe_allow_html=True)

st.markdown('<div class="title"><h1>NBA Match Predictor Using ML</h1></div>', unsafe_allow_html=True)

team1, middle, team2 = st.columns([2,2,2])


teams = ['ATL', 'BOS', 'BKN', 'CHA', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GSW', 'HOU', 'IND', 'LAC', 'LAL', 
        'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK', 'OKC', 'ORL', 'PHI', 'PHX', 'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS']

with team1: 
    st.markdown('<div class="title"><h1>Team 1</h1></div>', unsafe_allow_html=True)
    one = st.selectbox(label = "NBA Team #1:", options=teams)

with middle: 
     st.markdown('<div class="title"><h1>Match Date</h1></div>', unsafe_allow_html=True)
     a, mid, three = st.columns([1,1,1])
     with mid: 
        dat = st.date_input("Match Date:", value=datetime.date(2024, 6, 17))
     
with team2: 
     st.markdown('<div class="title"><h1>Team 2</h1></div>', unsafe_allow_html=True)
     two = st.selectbox(label = "NBA Team #2:", options=teams)

data = pd.read_csv('nba_games.csv')

st.write(data)

one, mid, two = st.columns([1,1,1])
with mid: 
    one, mid, two = st.columns([1,1,1])
    with mid:
        but = st.button("Generate Outcome")

data = data.sort_values("date")
data = data.reset_index(drop=True)
del data["mp.1"]
del data["mp_opp.1"]
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

ridge = RidgeClassifier(alpha=1)
split = TimeSeriesSplit(n_splits=3)
sfs = SequentialFeatureSelector(ridge, n_features_to_select=30, direction="forward", cv=split)

removed_columns = ["season", "date", "won", "target", "team", "team_opp"]
selected_columns = data.columns[~data.columns.isin(removed_columns)]

scaler = MinMaxScaler()
data[selected_columns] = scaler.fit_transform(data[selected_columns])
sfs.fit(data[selected_columns], data["target"])
predictors = list(selected_columns[sfs.get_support()])

data_rolling = data[list(selected_columns) + ["won", "team", "season"]]

def team_averages(team):
    numeric_cols = team.select_dtypes(include=['number']).columns
    rolling = team[numeric_cols].rolling(10).mean()
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

removed_columns_full = list(full.select_dtypes(include=['object']).columns) + removed_columns
selected_columns_full = full.columns[~full.columns.isin(removed_columns_full)]


full = full.dropna(subset=selected_columns_full)
full = full.dropna(subset=["target"])

sfs.fit(full[selected_columns_full], full["target"])

selected_features_mask = [col in predictors for col in full.columns]
selected_columns = full.columns[selected_features_mask]

if but:
    match_data = full[(full["team"] == one) & (full["team_opp_next"] == two) & (full["date_next"] == dat)]
    
    if match_data.empty:
        st.markdown('<div class="title"><h1>No data available for this matchup and date.</h1></div>', unsafe_allow_html=True)
    else:
        predictors = list(selected_columns)
        prediction = ridge.predict(match_data[predictors])[0]
        if prediction == 1:
            st.markdown(f'<div class="title"><h1>{one} won the game against {two} on {dat}.</h1></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="title"><h1>{two} won the game against {one} on {dat}.</h1></div>', unsafe_allow_html=True)



