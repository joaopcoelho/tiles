"""
TILEStools.py

Tools for the TILES project

"""
import numpy as np

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
def PRscore(y_true, y_pred):
    """
    Compute the PR-score of y_pred
    
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
assert np.isclose(PRscore([1, 1, 1], [1, 1, 1]), 1.)
assert np.isclose(PRscore([1, 1, 1], [0, 0, 0]), 0.)
assert np.isclose(PRscore([1,1,0,1], [1,0,0,0]), 0.5)
assert np.isclose(PRscore([1,1,0,1,0,0,1,0,1], [0,1,0,1,1,1,0,0,0]), 4./9.)

# define event_to_OHE function
def event_to_OHE(event_code, size):
    """
    *** PROBABLY DEPRECATED ***

    converts event_code to a One-Hot Encoded (OHE) vector of lenght size
    
    Input:
        event_code (int): the event code
        size (int): the lenght of the final OHE vector
    
    Output:
        event_OHE (ndarray): OHE vector
    
    Dependencies:
        numpy as np
    
    Comments:
        enforces event_code to be smaller than size.
        expecting 0 to be a valid event_code

    Examples:
        event_to_OHE(3, 5) = np.array([0,0,0,1,0])
        event_to_OHE(1, 2) = np.array([0,1])
    """
    import numpy as np
    
    assert event_code < size
    
    vec = np.zeros(size)
    vec[event_code] = 1.
    
    return vec

assert np.array_equal(event_to_OHE(4,5), np.array([0,0,0,0,1]))
assert np.array_equal(event_to_OHE(1,10), np.array([0,1,0,0,0,0,0,0,0,0]))
assert np.array_equal(event_to_OHE(0,2), np.array([1,0]))
