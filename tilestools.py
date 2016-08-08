"""
TILEStools.py

Tools for the TILES project

"""
import numpy as np
import pandas as pd

# generate GameID
def get_gameID(season,gametype,gamenum):
    """
    Generate gameID for stats.nba.com

    Input:
        season (int): season year in format YYYY
        gametype (str): either "R" for regular season or "P" for playoff
        gamenum (int): the game number

    Output:
        gameID (int): the requested game ID

    Comments:
        for playoff games, the game number should follow the rules outlined in DOCS.md

    """

    prefix = '00'


    se = str(season)[-2:]

    if gametype == "R":
        gt = str(2)
    elif gametype == "P":
        gt = str(4)
    else:
        raise ValueError("Unknown game type: {}".format(gametype))

    gn = str(gamenum).zfill(4)


    gameID = prefix + gt + se + '0' + gn

    return gameID

# quick tests
assert get_gameID(season=2015, gametype="R", gamenum=13)=='0021500013'
assert get_gameID(season=1999, gametype="P", gamenum=101)=='0049900101'
assert get_gameID(season=2013, gametype="R", gamenum=1230)=='0021301230'

# define PR-score
def F1score(y_true, y_pred):
    """
    Compute the F1-score of y_pred
    
    Input:
        y_true (array-like): ground truth value for labels
        y_pred (array_like): predicted values for labels
        
    Output:
        prscore (float): PR-score of y_pred
        
    Comments:
        PR-score combines precision and recall scores in order to create a single value to evaluate our model's performance
        PR-score = 2*P*R/(P+R), where P is precision and R is recall
        PR-score is always between 0 and 1, 0 is worse, 1 is best
    """
    
    import numpy as np
    from sklearn.metrics import precision_score, recall_score

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    prscore = 2.*precision*recall/(precision + recall)
    
    # account for precision, recall = 0.0
    if np.isnan(prscore):
        prscore = 0.0
        
    return prscore

# quick tests
assert np.isclose(F1score([1, 1, 1], [1, 1, 1]), 1.)
assert np.isclose(F1score([1, 1, 1], [0, 0, 0]), 0.)
assert np.isclose(F1score([1,1,0,1], [1,0,0,0]), 0.5)
assert np.isclose(F1score([1,1,0,1,0,0,1,0,1], [0,1,0,1,1,1,0,0,0]), 4./9.)

def build_df_events(df, features):
    """
    Builds a one-hot encoded dataframe from the original dataframe df and a list of relevant features
    
    Input:
        df (pandas dataframe): original dataframe with all events
        features (list): list of features to be retained
    
    Output:
        df_events (pandas dataframe): one-hot encoded dataframe
        
    Comments:
    
    """
    
    number_of_features = len(features)
    number_of_events = len(df)


    df_events = pd.DataFrame(data=np.zeros((number_of_events, number_of_features)), columns = features)

    # rebounds
    home_rebound_index = df[(df.EVENTMSGTYPE==4) & (df.HOMEDESCRIPTION.notnull())].index
    df_events.loc[home_rebound_index,"REBOUND_HOME"] = 1.0

    away_rebound_index = df[(df.EVENTMSGTYPE==4) & (df.VISITORDESCRIPTION.notnull())].index
    df_events.loc[away_rebound_index,"REBOUND_AWAY"] = 1.0

    # 2PT made
    idx = df[(df["EVENTMSGTYPE"]==1) & (df.HOMEDESCRIPTION.notnull()) & (df.HOMEDESCRIPTION.str.contains("^((?!3PT).)*$"))].index
    df_events.loc[idx,"2PT_MADE_HOME"] = 1.0

    idx = df[(df["EVENTMSGTYPE"]==1) & (df.VISITORDESCRIPTION.notnull()) & (df.VISITORDESCRIPTION.str.contains("^((?!3PT).)*$"))].index
    df_events.loc[idx,"2PT_MADE_AWAY"] = 1.0

    # 3PT made
    idx = df[(df.EVENTMSGTYPE==1) & (df.HOMEDESCRIPTION.notnull()) & (df.HOMEDESCRIPTION.str.contains("3PT"))].index
    df_events.loc[idx,"3PT_MADE_HOME"] = 1.0

    idx = df[(df.EVENTMSGTYPE==1) & (df.VISITORDESCRIPTION.notnull()) & (df.VISITORDESCRIPTION.str.contains("3PT"))].index
    df_events.loc[idx,"3PT_MADE_AWAY"] = 1.0

    # 2PT miss
    idx = df[(df["EVENTMSGTYPE"]==2) & (df.HOMEDESCRIPTION.str.contains("MISS")) & (df.HOMEDESCRIPTION.str.contains("^((?!3PT).)*$"))].index
    df_events.loc[idx,"2PT_MISS_HOME"] = 1.0

    idx = df[(df["EVENTMSGTYPE"]==2) & (df.VISITORDESCRIPTION.str.contains("MISS")) & (df.VISITORDESCRIPTION.str.contains("^((?!3PT).)*$"))].index
    df_events.loc[idx,"2PT_MISS_AWAY"] = 1.0

    # 3PT miss
    idx = df[(df["EVENTMSGTYPE"]==2) & (df.HOMEDESCRIPTION.str.contains("MISS")) & (df.HOMEDESCRIPTION.str.contains("3PT"))].index
    df_events.loc[idx,"3PT_MISS_HOME"] = 1.0

    idx = df[(df["EVENTMSGTYPE"]==2) & (df.VISITORDESCRIPTION.str.contains("MISS")) & (df.VISITORDESCRIPTION.str.contains("3PT"))].index 
    df_events.loc[idx,"3PT_MISS_AWAY"] = 1.0

    # turnovers
    idx = df[(df.EVENTMSGTYPE==5) & (df.HOMEDESCRIPTION.str.contains("Turnover"))].index
    df_events.loc[idx,"TOV_HOME"] = 1.0

    idx = df[(df.EVENTMSGTYPE==5) & (df.VISITORDESCRIPTION.str.contains("Turnover"))].index
    df_events.loc[idx,"TOV_AWAY"] = 1.0


    # timeouts
    idx = df[(df.EVENTMSGTYPE==9) & (df.HOMEDESCRIPTION.notnull())].index
    df_events.loc[idx,"TIMEOUT_HOME"] = 1.0

    idx = df[(df.EVENTMSGTYPE==9) & (df.VISITORDESCRIPTION.notnull())].index
    df_events.loc[idx,"TIMEOUT_AWAY"] = 1.0    


    return df_events

def build_cumulative_data(df_events, window_size=10):
    """
    Create a matrix of cumulative events 
    
    Input:
        df_events (pandas dataframe): a dataframe where every row is an event
        window_size (int): the size of the moving window
        
    Output:
        X (numpy matrix): the design matrix
        
    Comments:
        this is used as the design matrix and corresponding labels in the case of a moving-window approach
        the design matrix created, X, is just a sum over the previous events of whatever features df_events has
        X has size equal to df_events. If the number of previous events is smaller than window_size, just uses a smaller window (using all previous events)
    
    """
    # create design matrix X 

    # we define a window of size window_size and select every set of window_size points as a training example

    number_of_events, number_of_features = df_events.shape

    X = np.zeros((number_of_events, number_of_features))

    idx=0
    for event_id in range(number_of_events):
        before_events = df_events.iloc[max(0,event_id-window_size) : event_id]
        X[idx,:] = before_events.sum()
        idx+=1


    # remove last two columns in  X, which are the timeouts
    X = np.delete(X, np.s_[-2:], axis=1)  
    
    return X

def forward_window_labels(df_events, window_size=20):
    """
    Create a label vector with a forward window
    
    Input:
        df_events (pandas dataframe): a dataframe where every row is an event
        window_size (int): the size of the moving window
        
    Output:
        y (numpy array): the labels vector
    
    Comments:
        uses column TIMEOUT_HOME in df_events as input for the labels
        the label is whether the index of the next event is a timeout or not

    """

    number_of_events, number_of_features = df_events.shape

    timeout_events = list(df_events[df_events.TIMEOUT_HOME==1].index)

    windowed_timeouts = [k for event in timeout_events for k in range(event-window_size, event, 1)]

    # clean negative indices
    windowed_timeouts = [k for k in windowed_timeouts if k>=0]

    # create labels
    y = np.zeros(number_of_events)
    y[windowed_timeouts] = 1.0

    return y



def print_scores(model, X_train, y_train, X_test, y_test):
    """
    Compute scores for given model with training and test sets
    
    Input:
        model (sklearn.linear_model): the model with which to calculate scores
        X_train (numpy_array): training design matrix X
        y_train (numpy_array): training labels y
        X_test (numpy_array): test design matrix X
        y_test (numpy_array): test labels y
        
    Output:
        F1-score in test set
    
    Side Effects:
        prints the scores
        
    Comments:
        model must be fitted before calling this function
    
    """
    
    y_train_predicted = model.predict(X_train)
    y_test_predicted = model.predict(X_test)

    
    # accuracy scores
    print("Accuracy")
    print("Train: ", model.score(X_train,y_train))
    print("Test: ", model.score(X_test, y_test))
    print("\n")

    # use precision and recall metrics
    from sklearn.metrics import precision_score, recall_score

    precision_train = precision_score(y_train, y_train_predicted)
    recall_train = recall_score(y_train, y_train_predicted)

    precision_test = precision_score(y_test, y_test_predicted)
    recall_test = recall_score(y_test, y_test_predicted)

    print("Precision and Recall")
    print ("Train: ", precision_train, recall_train)
    print ("Test: ", precision_test, recall_test)
    print("\n")


    # F1 score
    from tilestools import F1score
    f1_train = F1score(y_train, y_train_predicted)
    f1_test = F1score(y_test, y_test_predicted)

    print("F1 score")
    print ("Train: ", f1_train)
    print ("Test: ", f1_test)

    
    return f1_test
