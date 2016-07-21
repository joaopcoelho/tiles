# Documentation for stats.nba.com API

This information was obtained by inspecting the playbyplay data of the following games:
* http://stats.nba.com/stats/playbyplayv2?EndPeriod=10&EndRange=55800&GameID=0041500314&RangeType=2&Season=2015-16&SeasonType=Playoffs&StartPeriod=1&StartRange=0

It is probably somewhat inaccurate and certainly incomplete, but it helps with having some notion of what the fields mean.
  
EVENTMSGACTIONTYPE:

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
