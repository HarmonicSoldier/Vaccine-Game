"""This module is responsible for defining strategy behaviour in the multiplayer control game.
Strategies, in this context, are behavioural strategies in the discretized version of the pandemic control game (i.e. they define a function from the set of possible histories to available actions).

To define a new type of strategy, define a module-level function named "strat_XXX", where "XXX" is the name you wish to identify the strategy by. This function should accept the following arguments: uid, country_list, expend_history, state_history, and any specific keyword arguments or parameters. The function should then return a recommended expenditure vector for the country of the specified uid for the provided history.

To define a strategy, create a Strategy object (or a subclass thereof) from a country using the provided constructor. Then, to use a strategy to compute an action for a country, call the get_choice() method, passing the history arrays as parameters. One can also change the type of a Strategy object using the set_strategy() method.
"""
import numpy as np
from functools import partial

def strat_default(uid, country_list, expend_history, state_history):
    """
    Strategy. Equivalent to a "default" strategy, with varying interpretations.
    Returns the result of the Idle strategy at the moment.

    Args:
        uid (:obj:`int`): the uid of the country which is to employ the strategy
        country_list(:obj:`list` of :obj:`Country`): list of country objects taking part in the game
        expend_history (:obj:`np.ndarray` of :obj:`float`): array of past expenditures (3D, end_time by num_countries by num_countries)
        state_history (:obj:`np.ndarray` of :obj:`float`): array of past states (3D, end_time + 1 by num_countries by 5)

    Returns:
        :obj:`np.ndarray` of :obj:`float`: recommended expenditure plan
    """

    return strat_idle(uid, country_list, expend_history, state_history)

def strat_idle(uid, country_list, expend_history, state_history):
    """
    Strategy. Equivalent to no strategy (default case)
    Returns a zero expenditure vector.

    Args:
        uid (:obj:`int`): the uid of the country which is to employ the strategy
        country_list(:obj:`list` of :obj:`Country`): list of country objects taking part in the game
        expend_history (:obj:`np.ndarray` of :obj:`float`): array of past expenditures (3D, end_time by num_countries by num_countries)
        state_history (:obj:`np.ndarray` of :obj:`float`): array of past states (3D, end_time + 1 by num_countries by 5)

    Returns:
        :obj:`np.ndarray` of :obj:`float`: recommended expenditure plan
    """
    num_countries = len(country_list)
    return np.zeros(shape=(num_countries,))

def strat_bang_greed(uid, country_list, expend_history, state_history):
    """
    Strategy. Bang-type strategy which is entirely selfish.
    Returns an expenditure vector which is max_consume at own position and zero elsewhere

    Args:
        uid (:obj:`int`): the uid of the country which is to employ the strategy
        country_list(:obj:`list` of :obj:`Country`): list of country objects taking part in the game
        expend_history (:obj:`np.ndarray` of :obj:`float`): array of past expenditures (3D, end_time by num_countries by num_countries)
        state_history (:obj:`np.ndarray` of :obj:`float`): array of past states (3D, end_time + 1 by num_countries by 5)

    Returns:
        :obj:`np.ndarray` of :obj:`float`: recommended expenditure plan
    """
    num_countries = len(country_list)
    self_pos = [country.uid for country in country_list].index(uid)
    expend_vect = np.zeros(shape=(num_countries,))
    expend_vect[self_pos] = country_list[self_pos].supp.max_consume
    return expend_vect

def strat_bang_all(uid, country_list, expend_history, state_history):
    """
    Strategy. Bang-type strategy which is cooperative.
    Returns an expenditure vector which is max_consume at own position and max_donate elsewhere

    Args:
        uid (:obj:`int`): the uid of the country which is to employ the strategy
        country_list(:obj:`list` of :obj:`Country`): list of country objects taking part in the game
        expend_history (:obj:`np.ndarray` of :obj:`float`): array of past expenditures (3D, end_time by num_countries by num_countries)
        state_history (:obj:`np.ndarray` of :obj:`float`): array of past states (3D, end_time + 1 by num_countries by 5)

    Returns:
        :obj:`np.ndarray` of :obj:`float`: recommended expenditure plan
    """
    self_pos = [country.uid for country in country_list].index(uid)
    self_country = country_list[self_pos]
    expend_vect = np.full(shape=(len(country_list),), fill_value=self_country.supp.max_donate)
    expend_vect[self_pos] = country_list[self_pos].supp.max_consume
    return expend_vect

def strat_combination(uid, country_list, expend_history, state_history, strat_list, strat_freq):
    """Strategy. A deterministic convex combination of strategies.
    Returns an expenditure vector which is a convex combination of strategies

    Args:
        uid (:obj:`int`): the uid of the country which is to employ the strategy
        country_list(:obj:`list` of :obj:`Country`): list of country objects taking part in the game
        expend_history (:obj:`np.ndarray` of :obj:`float`): array of past expenditures (3D, end_time by num_countries by num_countries)
        state_history (:obj:`np.ndarray` of :obj:`float`): array of past states (3D, end_time + 1 by num_countries by 5)
    	strat_list (:obj:`list` of :obj:`Strategy`): list of strategies to take combation over
        strat_freq (:obj:`np.ndarray` of :obj:`float`): vector of convex weights

    Returns:
        :obj:`np.ndarray` of :obj:`float`: recommended expenditure plan
    """

    num_strats = len(strat_list)
    expend_option_array = np.array(
        [strat.get_choice(uid, country_list, expend_history, state_history)
         for strat in strat_list])
    expend_vect = np.sum(expend_option_array, axis=1) * strat_freq

    return expend_vect

class Strategy:
    """Deterministic behavioural strategy.  Given a full history of prior actions, returns the expenditures for a provided country at the current time point

    For a list of currently available strategy options, please reference the "strat_" family of functions located elsewhere in the module. Strategies are named the same as their corresponding function, with the handle "strat_" removed from the start.

    Strategies are, by design able to be applied at any time to any country within the initialized country_list. The purpose of this is to allow multiple countries to use the same exact strategy object.
    
    Attributes:
        country_list (:obj:`tuple` of :obj:`Country`): tuple of references to country objects
        option (str): used to set strategy to one of a select presets
        args (dict): a dictionary of additional arguments to specify strategy
    """

    def __init__(self, country_list, option="default", **kwargs):

        self.country_list = tuple(country_list)
        self.num_countries = len(country_list)
        self.set_strategy(option, **kwargs)

    def set_strategy(self, option='default', **kwargs):
        """Selects a particular strategy from a predetermined list 

        The strategy function is stored internally and may be used through get_choice(...).

        Args:
            option (:obj:`str`): the type of strategy to be employed (see strat_ methods)
            **kwargs: additional arguments, specified as needed based on strategy

        Examples:
            >>> my_strategy.set_strategy('bang_all')
            >>> my_strategy.get_choice(expend_history, state_history)
        """
        match option:
            case 'idle':
                self.strategy = strat_idle
            case 'bang_greed':
                self.strategy = strat_bang_greed
            case 'bang_all':
                self.strategy = strat_bang_all
            case 'combination':
                self.strategy = partial(strat_combination, **kwargs)
            case _:
                self.strategy = strat_default

    def get_choice(self, uid, expend_history, state_history):
        """Based on chosen strategy, returns a suggestion of expenditures given a record of play until now.

        Suggested expenditures may not be properly scaled to either supply limitations or rate limitations.

        Args:
            uid (:obj:`int`): the uid of the country which is to employ the strategy
            expend_history (:obj:`np.ndarray` of :obj:`float`): array of past expenditures (3D, end_time by num_countries by num_countries)
            state_history (:obj:`np.ndarray` of :obj:`float`): array of past states (3D, end_time by num_countries by 5)

        Return:
            :obj:`np.ndarray` of :obj:`float`: choice of expenditures
        """
        return self.strategy(uid, self.country_list, expend_history, state_history)

class MixedStrategy:
    """A probablity distribution over a finite set of strategies.

       As a strategy, this strategy is identical to one of its provided possibilities at any given time. The particular strategy this is is determined (rolled) randomly at initialization and whenever the "reroll_strat" method is called. That is, at any given time between rerolls, calling the mixed strategy will return the same result. The probability of any particular strategy is determined upon rerolls by the strat_freq attribute.

    Attributes:
        strategy_list (:obj:`list` of :obj:`Strategy): a list of possible strategy objects
        strat_freqs (:obj:`np.ndarray` of :obj:`float): a numpy array of strategy frequencies
        num_strats (:obj:`int`): the number of strategies (need not provide)
        current_strat (:obj:`Strategy`): currently chosen strategy object
    """
    def __init__(self, strat_list, strat_freq):
        self.strat_list = strat_list
        self.strat_freq = strat_freq
        self.num_strats = len(self.strat_list)
        self.current_strat = None
        self.reroll_strat()

    def reroll_strat(self):
        """Based on provided probabilities, changes.
           Should be called whenever the "randomness" of the mixed strategy is to be incorporated.
           For instance, this could be either at every step in the simulation or at only the very beginning of the simulation.
        """
        rng = np.random.default_rng()
        self.current_strat = rng.choice(a=self.strat_list, p=self.strat_freq)

    def get_choice(self, uid, expend_history, state_history):
        """Based on chosen strategy, returns a suggestion of expenditures given a record of play until now.

        Suggested expenditures may not be properly scaled to either supply limitations or rate limitations.

        Args:
            uid (:obj:`int`): the uid of the country which is to employ the strategy
            expend_history (:obj:`np.ndarray` of :obj:`float`): array of past expenditures (3D, end_time by num_countries by num_countries)
            state_history (:obj:`np.ndarray` of :obj:`float`): array of past states (3D, end_time by num_countries by 5)

        Return:
            :obj:`np.ndarray` of :obj:`float`: choice of expenditures
        """
        return self.current_strat.get_choice(uid, expend_history, state_history)

class StrategyDistribution:
    """A frequency distribution of different strategies.
       Supports the adding of strategies and the updating of frequencies.

    Attributes:
        strat_list (:obj:`list` of :obj:`Strategy`): list of strategy objects to sample from
        strat_freq (:obj:`np.ndarray` of :obj:`float`): array of strategy frequencies
    """
    def __init__(self, strat_list, strat_freq):
        self.strat_list = strat_list
        self.strat_freq = strat_freq

    def add_strat(self, strat, pos=-1):
        """Appends a new strategy to the list of available strategies.
           Appended strategy has a frequency of zero by default

        Args:
            strat (:obj:`Strategy`): the strategy object which to append
            loc (:obj:`int`, optional): location within list which to append
        """
        self.strat_list.ins(pos, strat)
        self.strat_freq = np.insert(self.strat_freq, 0.0, pos)

    def get_num_strats(self):
        """Returns the number of strategies stored within the distribution"""
        return len(self.strat_list)

    def update_strat_freqs(self, new_freq):
        """Updates the frequencies of the constituent strategies to provided values

        Args:
            new_freq (:obj:`np.ndarray` of :obj:`float`): ordered list of new relative frequencies of strategies
        """
        self.strat_freq = new_freq

    def choose_random_strat(self):
        """Returns a random strategy according to the probability distributions therof

        Returns:
            :obj:`Strategy`: a strategy object, chosen randomly from the provided pool
        """
        rng = np.random.default_rng()
        return rng.choice(a=self.strat_list, p=self.strat_freq)
