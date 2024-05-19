import streamlit as st
import pandas as pd
import datetime 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score


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


teams = ['ATL', 'BOS', 'BRK', 'CHO', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GSW', 'HOU', 'IND', 'LAC', 'LAL', 
        'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK', 'OKC', 'ORL', 'PHI', 'PHO', 'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS']

with team1: 
    st.markdown('<div class="title"><h1>Team 1</h1></div>', unsafe_allow_html=True)
    one = st.selectbox(label = "NBA Team #1:", options=teams)

with middle: 
     st.markdown('<div class="title"><h1>Match Date</h1></div>', unsafe_allow_html=True)
     a, mid, c = st.columns([1,1,1])
     with mid: 
        dat = st.date_input("Match Date:", value=datetime.date(2024, 6, 17))
     
with team2: 
     st.markdown('<div class="title"><h1>Team 2</h1></div>', unsafe_allow_html=True)
     two = st.selectbox(label = "NBA Team #2:", options=teams)

@st.cache_data
def get_data():
    data = pd.read_csv('nba_games.csv', index_col = 0)
    data = data.sort_values("date").reset_index(drop=True)
    del data["mp.1"]
    del data["mp_opp.1"]    
    del data["index_opp"]
    return data

data = get_data()
st.write(data)

data["date"] = pd.to_datetime(data["date"])

data["team_opp_code"] = data["team_opp"].astype("category").cat.codes

data["target"] = (data["won"] == True).astype("int")


rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)

train = data[data["date"] < '2018-01-01']
test = data[data["date"] > '2018-01-01']

predictors = ["home", "team_opp_code", "fg%", "3p%", "ast", "stl", "blk", "ts%", "fg%_opp", "3p%_opp", "ast_opp", "stl_opp", "blk_opp", "ts%_opp" ]

rf.fit(train[predictors], train["target"])
preds = rf.predict(test[predictors])

combined = pd.DataFrame(dict(actual=test["target"], prediction=preds))

grouped_data = data.groupby(["team", "team_opp"])
def rolling_averages(group, cols, new_cols):
    group = group.sort_values("date")
    rolling_stats = group[cols].rolling(10, closed='left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)
    return group

cols = ['pts', 'orb', 'drb', 'ast', 'stl', 'blk', 'pts_opp', 'orb_opp', 'drb_opp', 'ast_opp', 'stl_opp', 'blk_opp']
new_cols = [f"{c}_rolling" for c in cols]

data_rolling = data.groupby(["team", "team_opp"]).apply(lambda x: rolling_averages(x, cols, new_cols))
data_rolling.index = range(data_rolling.shape[0])

def make_predictions(data, predictors):
    train = data[data["date"] < '2018-1-01']
    test = data[data["date"] > '2018-1-01']
    rf.fit(train[predictors], train["target"])
    preds = rf.predict(test[predictors])
    combined = pd.DataFrame(dict(actual=test["target"], prediction=preds), index=test.index)
    precision = precision_score(test["target"], preds)
    return combined, precision

combined, precision = make_predictions(data_rolling, predictors + new_cols)

combined = combined.merge( data_rolling[["date", "team", "team_opp", "won"]], left_index=True, right_index=True)

merged = combined.merge(combined, left_on=["date", "team"], right_on=["date", "team_opp"])

merged = merged.sort_values("date").reset_index(drop=True)

d, middy, e = st.columns([1,1,1])
with middy: 
    f, midd, g = st.columns([1,1,1])
    with midd:
        but = st.button("Generate Outcome")

if but: 
    matchup_exists = ((data["team"] == one) & (data["team_opp"] == two) | (data["team"] == two) & (data["team_opp"] == one)) & (data["date"] == pd.to_datetime(dat))

    if matchup_exists.any():

        team1_str = str(one)
        team2_str = str(two)
        specific_matchup_index = merged[(merged["team_x"] == team1_str) & (merged["team_opp_x"] == team2_str) & (merged["date"] == pd.to_datetime(dat))].index[0]
        specific_prediction = merged.loc[specific_matchup_index, "prediction_x"]

        if specific_prediction == 1:
            st.markdown(f'<div class="title"><h1>Based on the ML algorithm, {one} won the matchup against {two} on {dat}.</h1></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="title"><h1>Based on the ML algorithm, {two} won the matchup against {one} on {dat}.</h1></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="title"><h1>No matchup occured between {one} and {two} on {dat}.</h1></div>', unsafe_allow_html=True)

h, middl, i = st.columns([1,1,1])

with middl:
    st.write(merged)


    

    





