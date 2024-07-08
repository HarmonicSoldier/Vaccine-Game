"""This module is responsible for allowing one to construct evolutionary simulations to find evolutionary stable mixed strategy distributions amongst a pool of available strategies.

To construct an evolutionary simulation, complete the following steps:
1) Construct a Simulation object (see simulation module)
2) Create strategy distribution objects for each player
3) Determine the number of iterations and invoke EvolveSimulation class constructor

Evolutionary simulations can be played, iterated, and reset similar to classical Simulation objects.
They cannot be arbitrarily unwound, however

Examples:

    Constructs an evolutionary simulation with distinct pools of strategies

    >>> country_data1 = {"uid": 1, "name" = "Azkaban"}
    >>> country_data2 = {"uid": 2, "name" = "Bolivia"}
    >>> coop_coeffs = np.array([[1, 0.5], [-0.5, 1]])
    >>> my_simulation = Simulation([country_data1, country_data2], coop_coeffs, 100, 1)
    >>> country_list = my_simulation.country_list

    >>> strat1A = Strategy(country_list, "idle")
    >>> strat1B = Strategy(country_list, "bang_greed")
    >>> strat_freq = np.array([0.5, 0.5])
    >>> strat_dist1 = StrategyDistribution([strat1A, strat1B], strat_freq])

    >>> strat2A = Strategy(country_list, "idle")
    >>> strat2B = Strategy(country_list, "bang_greed")
    >>> strat_freq = np.array([0.5, 0.5])
    >>> strat_dist2 = StrategyDistribution([strat2A, strat2B], strat_freq])

    >>> contest_time = 10
    >>> my_evolve_simulation EvolveSimulation(simulation, [strat_dist1, strat_dist2], contest_time)
    >>> my_evolve_simulation.run_simulation()
    >>> my_evolve_simulation.reset_simulation()
    >>> my_evolve_simulation.iterate()

"""
import copy as copy
import functools as func
import itertools as iter
import numerics as num
import numpy as np
class EvolveSimulation:
    """ Simulates the evolution of different types of strategies for the pandemic evolution game.

    Controls for simulation are a simplified version of the controls for the pandemic simulation.
    Simulations can be ran, iterated, and reset. through corresponding methods.

    Attributes:
        simulation (:obj:`Simulation): used to run simulations to extract payoff information
        strat_dists (:obj:`list` of :obj:`StrategyDistribution`): for each country, a frequency distribution of strategies
        num_countries (:obj:`int`): the constituent number of countries being tracked
        max_num_strats (:obj:`int`): across all countries, the largest number of strategies within one distribution
        payoff_matrix (:obj:`np.ndarray` of :obj:`float`): matrix of payoffs for each strategy within each distribution (variable dimensions)
        contest_time (:obj:`int`): maxinum number of iterations for the simulation
        current_time (:obj:`int`): current number of completed iterations
    """

    def __init__(self, simulation, strat_dists, contest_time):

        #simulation objects, used to compute utilities
        self.simulation = copy.deepcopy(simulation)
        self.strat_dists = copy.deepcopy(strat_dists)

        #initialized simulation variables, used to track past distributions
        self.history = [-1] * (contest_time + 1)
        self.history[0] = copy.deepcopy(self.strat_dists)

        #quantitive variables, change only when countries/strategies are added/removed
        self.num_countries = self.simulation.num_countries
        self.max_num_strats = self.__compute_max_num_strats()

        #evolutionary variables, evolve over time
        self.__strat_freqs = self.__compute_strat_freqs()
        self.__payoff_matrix = self.__compute_payoff_matrix()
        self.__profile_freqs = self.__compute_profile_freqs()
        self.__fitness_scores = self.__compute_fitness_scores()

        #time variables, used to keep track of simulation time
        self.contest_time = contest_time
        self.current_time = 0

    def iterate(self, method="fittest", **kwargs):
        """Using the specified method, updates the frequency distributions of strategies.
           Updates occur internally and affect the strat_dists set of options.

        Args:
            method (:obj:`str`, optional): the type of method used for updating frequencies based on fitness scores
            kwargs: additional keyword arguments to supply to the corresponding iteration method
        """
        #only iterates if simulation has not reached the end
        if self.current_time < self.contest_time:
            self.update_distributions(method, **kwargs)
            self.__recompute()

            self.history[self.current_time + 1] = copy.deepcopy(self.strat_dists)
            self.current_time += 1

    def run_simulation(self, method="fittest", **kwargs):
        """Iterates through the evolutionary simulation until the ending frame
 
        Arguments:
            method (:obj:`str`, optional): the type of method used for updating frequencies
 
        """
        self.reset_simulation()
        for i in range(self.contest_time):
            self.iterate(method, **kwargs)
 
    def reset_simulation(self):
        """Resets the evolutionary simulation to the initialized state """
        self.current_time = 0
        self.strat_dists = self.history[0]

    def __recompute(self):
        """Given current strategy distributions, internally recomputes relevant depenedent quantities.
        Should be called whenever strategy distributions are updated.

        Following a change in strategy frequencies, recomputes the following attributes internally: max_num_strats, payoff_matrix, profile_freqs, fitness_scores
        """
        self.__max_number_strats = self.__compute_max_num_strats()
        self.__strat_freqs = self.__compute_strat_freqs()
        self.__payoff_matrix = self.__compute_payoff_matrix()
        self.__profile_freqs = self.__compute_profile_freqs()
        self.__fitness_scores = self.__compute_fitness_scores()

    def __compute_max_num_strats(self):
        """Given all distributions, computes the maximum number of strategies across distributions.

        Returns:
            max_num_strats (:obj:`int`): integer number of encoding maximum number of strats possessed by a distribution
        """
        return max((dist.get_num_strats() for dist in self.strat_dists))

    def __compute_strat_freqs(self):
        """Given all distributions, computes a shared numpy array storing all of them

        Returns:
            strat_freqs (:obj:`np.ndarray` of :obj:`float`): array of strategy frequencies for each strategy (2D, num_countries by max_num_strats)
        """
        num_strats = [dist.get_num_strats() for dist in self.strat_dists]
        strat_freqs = np.zeros(shape=(self.num_countries, max(num_strats)))
        for i in range(self.num_countries):
            strat_freqs[i, 0:num_strats[i]] = self.strat_dists[i].strat_freq

        return strat_freqs

    def __compute_payoff_matrix(self):
        """Given all strategies within all distributions, computes a matrix of payoffs.
           Registers payoff matrix internally 

        Returns:
           payoff_matrix(:obj:`np.ndarray` of :obj:`float`): variable dimension array.
                given indices i1, i2, ... in, the value payoff_matrix[i1, i2, ..., in, :]
                is a vector of payoffs for each country.
        """
        #constructs a tuple of array dimension
        dims = [dist.get_num_strats() for dist in self.strat_dists] + [self.num_countries,]

        #builds the payoff matrix by computing each case independently
        #'profile', in this context, refers to a selection of strategies by number

        #auxillary functions: sets appropriate strategies
        def set_strats(profile):
            for i in range(self.num_countries):
                uid = self.simulation.uid_list[i]
                strat = self.strat_dists[i].strat_list[profile[i]]
                self.simulation.replace_strategy(uid, strat)

        #auxillary function; computes payoffs for set strategies
        def get_payoffs():
            self.simulation.run_simulation()
            return self.simulation.get_payoffs()

        payoff_matrix = np.empty(shape=dims)
        for profile in iter.product(*[range(dim) for dim in dims]):
            set_strats(profile)
            payoffs = get_payoffs()
            payoff_matrix[profile, ...] = payoffs

        self.simulation.reset_simulation()
        return payoff_matrix

    def __compute_profile_freqs(self):
        """Computes a matrix of probabilities for all possible strategy profiles

        Returns:
            profile_freqs(:obj:`np.ndarray` of :obj`float): variable dimension array.
            given indices i1, i2, ... in, the value payoff_matrix[i1, i2, ..., in]
            is an absolute probability of occurence
        """

        strat_freq_list = [dist.strat_freq for dist in self.strat_dists]

        #look up documentation for refresher on what this does
        print(strat_freq_list)
        return func.reduce(np.multiply, np.ix_(*strat_freq_list))

    def __compute_fitness_scores(self):
        """Returns an array of each fitness values for each strategy in each distribution.
           Currently, fitness scores are calculated as weighted average of payoffs.

        Returns:
          :obj:`np.ndarray` of :obj:`float`: for each strategy distribution, a an array of fitness values for each constitutent strategy (2D, num_countries by max_num_strats)

        Notes:
            for accurate results, call after updating payoff_matrix, profile_freqs, and max_num_strats
        """

        fitness_scores = np.zeros(shape=(self.num_countries, self.__compute_max_num_strats()))

        for i in range(self.num_countries):
            util_matrix = self.__profile_freqs * self.__payoff_matrix[..., i]
            fitness_vect = num.safe_divide(
                array1=np.sum(util_matrix, axis=i),
                array2=self.strat_dists[i].strat_freq,
                zero_substitute=0.0)

            num_strats = self.strat_dists[i].get_num_strats()
            fitness_scores[i, 0:num_strats] = fitness_vect

        return fitness_scores
            
    def update_distributions(self, method="fittest", **kwargs):
        """Assuming updated state, updates internal frequencies of strategies according to provided method. 
           
        Arguments:
            method (str): the rule used to update distributions

            See update_distribution_XXXX family of methods for options.
            If invalid option is presented
        """
        match method.lower():
            case "fittest":
                self.update_distributions_fittest(kwargs["n"])
            case _:
                pass

    def update_distributions_fittest(self, n):
        """Keyword is "fittest". Updates distributions on the basis of survival of the fittest

        1) lowest n% of frequencies are cut, and the remaining populations are scaled up to fit
        2) randomly reproduces existing individuals to fill remaining frequencies

        Args:
            n (:obj:`float`): percent (out of 100) of least fit individuals to drop
            fitness_scores (:obj:`np.ndarray` of :obj:`float`): array of evaluated fitness scores (2D, num_countries by max_num_strats)
        """

        #computes how many strategies each player has
        num_strats = [dist.get_num_strats() for dist in self.strat_dists]

        #first, drops the lowest scoring strategies of each country
        for i in range(self.num_countries):

            #defines a desired distribution
            strat_dist = self.strat_dists[i]
            fittness_vect = self.__fitness_scores[i, 0:strat_dist.get_num_strats()]
            
            #initializes information about what to drop    
            left_to_drop = n / 100.0
            drop_amounts = np.zeros_like(fittness_vect)

            #determines order of strategies to drop
            sorted_strats_index = sorted(
                range(num_strats[i]),
                key=lambda j: fittness_vect[j]
            )

            #drops largest amount from each before ending
            for j in range(num_strats[i]):
                idx = sorted_strats_index[j]
                drop_amounts[idx] = min(left_to_drop, strat_dist.strat_freq[idx])
                left_to_drop -= drop_amounts[idx]

            strat_dist.strat_freq -= drop_amounts

        #then scales the frequencies to reach one
        for i in range(self.num_countries):
            strat_dist = self.strat_dists[i]
            strat_dist.strat_freq /= np.sum(strat_dist.strat_freq)
