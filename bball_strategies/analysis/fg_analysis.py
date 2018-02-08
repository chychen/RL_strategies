"""
Ref: http://buckets.peterbeshai.com/app/#/leagueView/2015?l_countMin=1&l_showLegend=false&p_countMin=1
Source: view-source:http://buckets.peterbeshai.com/app/#/leagueView/2015?l_countMin=1&l_showLegend=false&p_countMin=1 
1. fetch all 2015-2016 season from Source above. (save it as numpy)
2. implement the function 'get_prob_by_loc()'
"""


def get_prob_by_loc(x, y, mode='FQ'):
    """ ### Get Probability By Location

    Args
    ----
    x : ndarray, dtype=float32, shape=(None,), range=[0, 94.0)
    y : ndarray, dtype=float32, shape=(None,), range=[0, 50.0)
    mode : str,
        - 'FQ' : Shoot Frequency
        - 'FG' : Field Goal Rate
    
    Raises
    ------
    ValueError : if input argument is not valid.

    Return
    ------
    prob : ndarray, dtype=float32, shape=(None,), range=[0, 1.0)
        given location, returns the probability of FQ/FG.
    """
    raise NotImplementedError()


def main():
    get_prob_by_loc(x=0.1, y=0.1, mode)

if __name__ == '__main__':
    main()