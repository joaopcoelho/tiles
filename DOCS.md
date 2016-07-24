# Documentation for stats.nba.com API

This information was obtained by inspecting the playbyplay data of the following games:
* http://stats.nba.com/stats/playbyplayv2?EndPeriod=10&EndRange=55800&GameID=0041500314&RangeType=2&Season=2015-16&SeasonType=Playoffs&StartPeriod=1&StartRange=0

It is probably somewhat inaccurate and certainly incomplete, but it helps with having some notion of what the fields mean.

### Game IDs

Game IDs follow this logic:

GameID = *00Syy0xxxx*

where:

- S is either 2 for regular season or 4 for playoff games
- yy is the season (e.g. 15 for 2015/2016, 99 for 1999/2000)
- xxxx is the game number

About the game number:

- for regular season games, this is just the game number. It's a value between 1 and 1230 (in case of a normal 82-game season with 30 teams); in general, it's a value between 1 and G*T/2, where G is games per team and T is total number of teams.

- for playoff games, this is a code of the form 0RMG, where 0 is just a 0, R is the playoff round (normally 1-4, although probably less in past ages), M is the matchup number (varying between *0* and the total number of matchups for that playoff round, for both conferences) and G is the game number within that particular matchup (e.g. 1 for Game 1, 6 for Game 6).

Examples:

GameID=0021500001: regular season, 2015-2016, game #1 (Detroit Pistons 106 - 94 Atlanta Hawks)
GameID=0021501230: regular season, 2015-2016, game #1230 (Denver Nuggets 99 - 107 Portland Trail Blazers)
GameID=0021000873: regular season, 2010-2011, game #873 (Dallas Mavericks 105 - 99 Washington Wizards)
GameID=0041500154: playoffs, 2015-2016, round 1, matchup 5, game 4 (San Antonio Spurs 116 - 95 Memphis Grizzlies)
GameID=0041200403: playoffs, 2012-2013, round 4, matchup 0, game 3 (Miami Heat 77 - 103 San Antonio Spurs)
GameID=0029700234: regular season, 1997-1998, game #234 (Toronto Raptors 98 - 115 Utah Jazz)
GameID=0026300101: regular season, 1963-1964, game #101 (New York Knicks 102 - 116 Baltimore Bullets)


### EVENTMSGTYPE

1: Made shots

2: Miss shots

3: Free Throws

4: Rebounds

5: Turnovers

6: Fouls

8: Subs

9: Timeouts

10: Jump Ball

12: Start of quarter

13: End of quarter  

### EVENTMSGACTIONTYPE:

0: Rebound, Sub, Jump Ball

1: Jump Shot, Bad Pass Turnover, Timeout Regular, P.Foul

2: Lost Ball Turnover, Timeout Short, S.Foul (shooting foul?)

3: Hook Shot, L.B.FOUL 

4: offensive fouls

5: Layup, Foul Turnover 

7: dunks

11: free throws - 1st of 2

12: free throws - 2nd of 2

13: free throws - 1st of 3

14: free throws - 2nd of 3

15: free throws - 3rd of 3

28: personal take foul

37: offensive foul turnover

39: step out of bounds turnover

40: Out of Bounds Lost Ball Turnover

42: Driving Layup

43: Alley-Oop

44: Reverse layup

45: Out of Bounds - Bad Pass Turnover Turnover

47: Turnaround Jump Shot

52: Alley-Oop Dunk

58: Turnaround Hook Shot 

72: Putback layup

73: Driving reverse layup

76: Running Finger Roll Layup

78: Floating Jump Shot

79: Pullup Jump Shot

80: Step Back Jump Shot

83: Fadeaway Bank Shot

87: Putback Dunk

97: Tip Layup

98: Cutting Layup Shot

101: Driving Floating Jump Shot

102: Driving Floating Bank Jump Shot

108: Cutting Dunk Shot
