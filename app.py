#import libraries and packages
import streamlit as st
import pandas as pd
import datetime 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

#setting pg layout and title of tab
st.set_page_config(page_title="NBA Matchup Predictor", layout="wide")

#creating gradient background using css styling
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

#creating title style usin css styling
title = """
<style>
.title {
    text-align: center;
}
</style>
"""

#adding components to website
st.markdown(gradient, unsafe_allow_html=True)
st.markdown(title, unsafe_allow_html=True)

#title of tab
st.markdown('<div class="title"><h1>NBA Match Predictor Using ML</h1></div>', unsafe_allow_html=True)

#column formatting - creating three columns in the wbsite
team1, middle, team2 = st.columns([2,2,2])

#all team names
teams = ['ATL', 'BOS', 'BRK', 'CHO', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GSW', 'HOU', 'IND', 'LAC', 'LAL', 
        'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK', 'OKC', 'ORL', 'PHI', 'PHO', 'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS']

#column 1 
with team1: 

    #drop_down for team 1
    st.markdown('<div class="title"><h1>Team 1</h1></div>', unsafe_allow_html=True)
    one = st.selectbox(label = "NBA Team #1:", options=teams)

#column 2
with middle: 
     
     #drop_down for date
     st.markdown('<div class="title"><h1>Match Date</h1></div>', unsafe_allow_html=True)
     
     #center col for date input
     a, mid, c = st.columns([1,1,1])
     with mid: 
        dat = st.date_input("Match Date:", value=datetime.date(2024, 6, 17))

#column 3     
with team2:

     #drop down for team 2
     st.markdown('<div class="title"><h1>Team 2</h1></div>', unsafe_allow_html=True)
     two = st.selectbox(label = "NBA Team #2:", options=teams)


@st.cache_data
def get_data():

    #read dataset
    data = pd.read_csv('nba_games.csv', index_col = 0)

    #sort dataset based on date
    data = data.sort_values("date").reset_index(drop=True)

    #delete extra columns 
    del data["mp.1"]
    del data["mp_opp.1"]    
    del data["index_opp"]
    return data

#call func
data = get_data()

view_data = data[data["date"] >= "2018-01-01"]

view_data = view_data.drop(columns=('won'))



#display data
st.write(view_data)

#changing date column value from string to datetime 
data["date"] = pd.to_datetime(data["date"])

#adding numerical column for opposite team names 
data["team_opp_code"] = data["team_opp"].astype("category").cat.codes

#adding target column that translates what team won in numbers
data["target"] = (data["won"] == True).astype("int")


# initialize RandomForestClassifier
rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)

#train var - data from before 2018
train = data[data["date"] < '2018-01-01']

#test var - data from after 2018
test = data[data["date"] >= '2018-01-01']

#predictor variables (column names that would be helpful to predict outcome)
predictors = ["home", "team_opp_code", "fg%", "3p%", "ast", "stl", "blk", "ts%", "fg%_opp", "3p%_opp", "ast_opp", "stl_opp", "blk_opp", "ts%_opp" ]

#fitting data with ForestClassifier
rf.fit(train[predictors], train["target"])

#creating predictions
preds = rf.predict(test[predictors])

#creating a data frame combining actual wins with predicted wins in certain matchups
combined = pd.DataFrame(dict(actual=test["target"], prediction=preds))

#create a data set that groups by team and team opponenet to create a neater dataset
grouped_data = data.groupby(["team", "team_opp"])


def rolling_averages(group, cols, new_cols):
    #sort individual matchups by dates
    group = group.sort_values("date")

    #gather the average data of the 10 previous games of the specific matchup
    rolling_stats = group[cols].rolling(10, closed='left').mean()
    group[new_cols] = rolling_stats

    #drop null data
    group = group.dropna(subset=new_cols)
    return group

#columns that will be used in getting the averages for
cols = ['pts', 'orb', 'drb', 'ast', 'stl', 'blk', 'pts_opp', 'orb_opp', 'drb_opp', 'ast_opp', 'stl_opp', 'blk_opp']

#add new cols based on rolling data
new_cols = [f"{c}_rolling" for c in cols]

#call func for each matchup in the data dataset
data_rolling = data.groupby(["team", "team_opp"]).apply(lambda x: rolling_averages(x, cols, new_cols))

data_rolling.index = range(data_rolling.shape[0])

#making better predictions 
def make_predictions(data, predictors):

    #get the train and test datasets
    train = data[data["date"] < '2018-1-01']
    test = data[data["date"] > '2018-1-01']

    #fit data with ForestClassifier
    rf.fit(train[predictors], train["target"])

    #assign value of predictions
    preds = rf.predict(test[predictors])

    #create a dataset that combines the actual outcome of the matchups vs the predicted outcome
    combined = pd.DataFrame(dict(actual=test["target"], prediction=preds), index=test.index)

    #get precision score of predictor
    precision = precision_score(test["target"], preds)
    return combined, precision

#calling func
combined, precision = make_predictions(data_rolling, predictors + new_cols)

#merging data with data_rolling columns 
combined = combined.merge( data_rolling[["date", "team", "team_opp", "won"]], left_index=True, right_index=True)

#adding opponent data to combined data
merged = combined.merge(combined, left_on=["date", "team"], right_on=["date", "team_opp"])

#sort merged data based on date
merged = merged.sort_values("date").reset_index(drop=True)

#add extra column to get the matchups in alphabetical order with the date
merged['team_combo'] = merged.apply(lambda x: '-'.join(sorted([x['team_x'], x['team_opp_x']])) + '-' + str(x['date'].date()), axis=1)

#delete any duplicates
merged = merged.drop_duplicates(subset=['team_combo'], keep='first')



#creating button
d, middy, e = st.columns([1,1,1])
with middy: 
    f, midd, g = st.columns([1,1,1])
    with midd:
        but = st.button("Generate Outcome")

#if clicked 
if but: 

    #assign input values
    team_one = one
    team_two = two
    date_input = pd.to_datetime(dat)

    
    #if match exists in merged dataset
    matchup_exist = (((merged["team_x"] == team_one) & (merged["team_opp_x"] == team_two)) | ((merged["team_y"] == team_two) & (merged["team_opp_y"] == team_one))) & \
                    (merged["date"] == date_input)


    if matchup_exist.any():

        #get specific matchup
        specific_matchup = merged[matchup_exist]

        #if it contains data
        if not specific_matchup.empty:

            #get to the prediction_x col for that match
            specific_matchup_index = specific_matchup.index[0]
            specific_prediction = specific_matchup.loc[specific_matchup_index, "prediction_x"]

            #if team 1 won
            if specific_prediction == 1:
                st.markdown(f'<div class="title"><h1>Based on the ML algorithm, {one} won the matchup against {two} on {dat}.</h1></div>', unsafe_allow_html=True)
                
            #if team 2 won
            else:
                st.markdown(f'<div class="title"><h1>Based on the ML algorithm, {two} won the matchup against {one} on {dat}.</h1></div>', unsafe_allow_html=True)
             
        #no match occured
        else:
            st.markdown(f'<div class="title"><h1>Match prediction is unavailable. Missing data of the matchup occured between {one} and {two} on {dat}.</h1></div>', unsafe_allow_html=True)
     
        #no match occured
    else:
        st.markdown(f'<div class="title"><h1>Match prediction is unavailable. Missing data of the matchup occured between {one} and {two} on {dat}.</h1></div>', unsafe_allow_html=True)


h, middl, i = st.columns([1,1,1])

#print out merged dataset to show to user
with middl:
    st.write(merged)
    

    





