"""This module is responsible for defining strategy behaviour in the multiplayer control game.
Strategies, in this context, are behavioural strategies in the discretized version of the pandemic control game (i.e. they define a function from the set of possible histories to available actions).

To define a strategy, create a Strategy object (or a subclass thereof) from a country using the provided constructor. Then, to use a strategy to compute an action for a country, call the get_choice() method, passing the history arrays as parameters.

Specific strategies, potentially with suppliable parameters, are defined with the "strat_" family of methods, which are not contained within any particular class. These methods are publically available, but they should generally not be used.
"""
import numpy as np
from functools import partial

def strat_default(uid, country_list, expend_history, state_history):
    """
    Strategy. Equivalent to a "default" strategy, with varying interpretations.
    Returns the result of applying a default strategy.

    Args:
        uid (:obj:`int`): the uid of the country which is to employ the strategy
        country_list(:obj:`list` of :obj:`Country`): list of country objects taking part in the game
        expend_history (:obj:`np.ndarray` of :obj:`float`): array of past expenditures (3D, end_time by num_countries by num_countries)
        state_history (:obj:`np.ndarray` of :obj:`float`): array of past states (3D, end_time + 1 by num_countries by 5)
        num_countries (int): number of countries under consideration

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
        num_countries (int): number of countries under consideration

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
    expend_vect = np.full(shape=(len(country_list),), fill_value=country.supp.max_donate)
    expend_vect[self_pos] = country_list[self_pos].supp.max_consume
    return expend_vect

def strat_combination(uid, country_list, expend_history, state_history, strat_list, strat_weight):
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
    expend_vect = np.zeros(shape=(len(country_list), ))

    for i in range(num_strats):
        strt = strat_list[i]
        wght = strat_weight[i]
        expend_vect += wght * strt.get_choice(uid, country_list, expend_history, state_history)

    return expend_vect

class Strategy:
    """Deterministic behavioural strategy.  Given a full history of prior actions, returns the expenditures for a provided country at the current time point

    For a list of currently available strategy options, please reference the "strat_" family of functions located elsewhere in the module. Strategies are named the same as their corresponding function, with the handle "strat_" removed from the start.

    Strategies are, by design able to be applied at any time to any country within the initialized country_list. The purpose of this is to allow multiple countries to use the same exact strategy object.
    
    Attributes:
        country_list (:obj:`list` of :obj:`Country`): list of references to country objects
        option (str): used to set strategy to one of a select presets
        args (dict): a dictionary of additional arguments to specify strategy
    """

    def __init__(self, country_list, option="default", kwargs={}):

        self.country_list = country_list
        self.num_countries = len(country_list)
        self.set_strategy(option, kwargs)

    def set_strategy(self, option='default', kwargs={}):
        """Selects a particular strategy from a predetermined list 

        The strategy function is stored internally and may be used through get_choice(...).

        Args:
            option (:obj:`str`): the type of strategy to be employed (see strat_ methods)
            kwargs (dict): a string-value dictionary of additional arguments for strategy (varies based on implementation)

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
                self.strategy = partial(strat_combination, strat_list=kwargs["strat_list"], strat_weight=kwargs["strat_weight"])
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

class MixedStrategy(Strategy):
    """A probablity distribution over a finite set of strategies.

       As a strategy, this strategy is identical to one of its provided possibilities at any given time. The particular strategy this is is determined (rolled) randomly at initialization and whenever the "reroll_strat" method is called. The probability of any particular strategy is determined by the strat_freq method.

    Attributes:
        strategy_list (:obj:`list` of :obj:`Strategy): a list of possible strategy objects
        strat_freqs (:obj:`np.ndarray` of :obj:`float): a numpy array of strategy frequencies
        num_strats (:obj:`int`): the number of strategies (need not provide)
        current_strat (:obj:`Strategy`):
    """
    def __init__(strat_list, strat_freq):

        self.strat_list = strat_list
        self.strat_freq = strat_freq
        self.num_strats = len(self.strat_list)
        self.current_strat = None
        self.reroll_strat()

    def reroll_strat():
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
        return self.current_strat(self, uid, expend_history, state_history)

class StrategyDistribution:
    """A frequency distribution of different strategies.

    Attributes:
        strat_list (:obj:`list` of :obj:`Strategy`): list of strategy objects to sample from
        strat_freq (:obj:`np.ndarray` of :obj:`float`): array of strategy frequencies
    """
    def __init__(strat_list, strat_freq):
        self.strat_list = strat_list
        self.strat_freq = strat_freq
        self.num_strats = len(self.strat_list)

    def choose_random_strat():
        """Returns a random strategy according to the probability distributions therof

        Returns:
            :obj:`Strategy`: a strategy object, chosen randomly from the provided pool
        """
        rng = np.random.default_rng()
        return rng.choice(a=self.strat_list, p=self.strat_freq)

    def update_strat_freqs(new_freq):
        """Updates the frequencies of the constituent strategies to provided values

        Args:
            new_freq (:obj:`np.ndarray` of :obj:`float`): ordered list of new relative frequencies of strategies
        """
        self.strat_freq = new_freq