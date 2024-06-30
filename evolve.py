
from simulation import Simulation
from strategy import Strategy, StrategyDistribution

from copy import deepcopy
import itertools as iter
import numerics as num

class EvolveSimulation:
    """ Simulates the evolution of different types of strategies for the pandemic evolution game.

    Controls for simulation are a simplified version of the controls for the pandemic simulation.
    Simulations can be ran, iterated, and reset. through corresponding methods.

    Attributes:
        simulation (:obj:`Simulation): used to run simulations upon
        strat_dists (:obj:`list` of :obj:`StrategyDistribution`): for each country, a frequency distribution of strategies
        num_countries (:obj:`int`): the constituent number of countries being tracked
        max_num_strats (:obj:`int`): across all countries, the largest number of strategies within one distribution
        payoff_matrix (:obj:`np.ndarray` of :obj:`float`): matrix of payoffs for each strategy within each distribution (variable dimensions)
        contest_time (:obj:`int`): maxinum number of iterations for the simulation
        current_time (:obj:`int`): current number of completed iterations
    """

    def __init__(simulation, strat_dists, contest_time):

        #simulation objects, used to compute utilities
        self.simulation = copy.deepcopy(simulation)
        self.strat_dists = copy.deepcopy(strat_dists)

        #initialized simulation variables, used to 
        self.__init_simulation = copy.deepcopy(self.simulation)
        self.__init_strat_dists = copy.deepcopy(self.strat_dists)

        #quantitive variables, change only when countries/strategies are added/removed
        self.num_countries = self.simulation.num_countries
        self.max_num_strats = self.__compute_max_num_strats()

        #evolutionary variables, evolve over time
        self.strat_freqs = self.__compute_strat_freqs()
        self.payoff_matrix = self.__compute_payoff_matrix()
        self.profile_freqs = self.__compute_profile_freqs()
        self.fitness_scores = self.__compute_fitness_scores()

        #time variables, used to keep track of simulation time
        self.contest_time = contest_time
        self.current_time = 0

    def iterate(method="fittest", get_result=False, **kwargs):
        """Updates the evolutionary population frequencies

        Args:
            get_result (:obj:`bool`, optional): if true, returns an array of strategy frequencies for each country's distribution
            method (:obj:`str`, optional): the type of method used for updating frequencies based on fitness scores
            kwargs: additional keyword arguments to supply to the corresponding iteration method

        Returns:
            new_dist_freqs (:obj:`np.ndarray` of :obj:`float): if requested; ordered array of strategy frequencies (2D, num_countries by max_num_strats)
        """

        #termination condition, ensures end-of-time condition is satisfied
        if self.current_time >= self.contest_time:
            return None

        #computes fittness scores for each strategy, then computes new frequencies
        self.fitness_scores = self.__compute_fitness_scores()
        self.update_distribution(method, **kwargs)

        #increments the time of the contest by one
        self.contest_time += 1
        return self.__compute_strat_freqs() if get_result else None

    def run_simulation(method="fittest", get_history=False, **kwargs):
        """Iterates through the evolutionary simulation until the ending frame
 
        Arguments:
            method (:obj:`str`, optional): the type of method used for updating frequencies
            get_history (:obj:`bool`, optional): if true, returns a full history of strategy distributions
 
        Returns:
            history (:obj:`np.ndarray` of :obj:`float`): if specified. array of strategy frequencies for each player and time (3D, contest_time by num_countires by max_num_strats)
        """
        self.reset_simulation()
 
        past_frames = []
        for i in range(self.contest_time):
            frame = self.iterate(method, get_history, **kwargs)
            if get_history: past_frames.append(frame)
 
        return np.array(past_frames) if get_history else None

    def reset_simulation():
        """Resets the evolutionary simulation to the initialized state """
        self.current_time = 0
        self.simulation = copy.deepcopy(self.__init_simulation)
        self.strat_dists = copy.deepcopy(self.__init_strat_dists)

    def refresh():
        """Following a change in strategy frequencies, recomputes the following attributes internally: max_num_strats, payoff_matrix, profile_freqs, fitness_scores
        """
        self.max_number_strats = compute_max_num_strats()
        self.strat_freqs = self.__compute_strat_freqs()
        self.payoff_matrix = self.__compute_payoff_matrix()
        self.profile_freqs = self.__compute_profile_freqs()
        self.fitness_scores = self.__compute_fitness_scores()

    def __compute_max_num_strats():
        """Given all distributions, computes the maximum number of strategies across distributions.

        Returns:
            max_num_strats (:obj:`int`): integer number of encoding maximum number of strats possessed by a distribution
        """
        return max((dist.num_strats for dist in self.strat_dists))

    def __compute_strat_freqs():
        """Given all distributions, computes a shared numpy array storing all of them

        Returns:
            strat_freqs (:obj:`np.ndarray` of :obj:`float`): array of strategy frequencies for each strategy (2D, num_countries by max_num_strats)
        """
        num_strats = (dist.num_strats for dist in self.strat_dists)
        strat_freqs = np.zeros(shape=(self.num_countries, max(num_strats)))

        for i in range(self.num_countries):
            strat_freqs[i, 0:num_strats[i]] = self.strat_dists[i].strat_freq
        return strat_freqs

    def __compute_payoff_matrix():
        """Given all strategies within all distributions, computes a matrix of payoffs.
           Registers payoff matrix internally 

        Returns:
           payoff_matrix(:obj:`np.ndarray` of :obj:`float`): variable dimension array.
                given indices i1, i2, ... in, the value payoff_matrix[i1, i2, ..., in, :]
                is a vector of payoffs for each country.
        """
        #constructs the number of dimensions,
        dims = (dist.num_strats for dist in self.strat_dists) + (self.num_countries,)

        #builds the payoff matrix by computing each case independently
        #'profile', in this context, refers to a selection of strategies by number
        payoff_matrix = np.zeros(shape=dims)
        for profile in iter.product(*dims):
            simulation.reset_simulation()

            for i in range(self.num_countries):
                uid = self.simulation.uid_list[i]
                strat = self.strat_dists[i].strat_list[profile[i]]
                self.simulation.replace_strategy(uid, strat)

            simulation.run_simulation()
            payoffs = simulation.get_payoffs()
            payoff_matrix[profile] = payoffs

        simulation.reset_simulation()
        return payoff_matrix

    def __compute_profile_freqs():
        """Computes a matrix of probabilities for all possible strategy profiles

        Returns:
            profile_freqs(:obj:`np.ndarray` of :obj`float): variable dimension array.
            given indices i1, i2, ... in, the value payoff_matrix[i1, i2, ..., in]
            is an absolute probability of occurence
        """

        strat_freq_list = [self.dist.strat_freq for dist in self.strat_dists]

        #look up documentation for refresher on what this does
        return np.multiply.reduce(np.ix_(*strat_freq_list))

    def __compute_fitness_scores():
        """Returns an array of each fitness values for each strategy in each distribution.
           Currently, fitness scores are calculated as weighted average of payoffs.

        Returns:
          :obj:`np.ndarray` of :obj:`float`: for each strategy distribution, a an array of fitness values for each constitutent strategy (2D, num_countries by max_num_strats)

        Notes:
            for accurate results, call after updating payoff_matrix, profile_freqs, and max_num_strats
        """

        max_num_strats = self.max_num_strats
        payoff_matrix = self.payoff_matrix
        profile_freqs = self.profile_freqs

        for i in range(self.num_countries):
            util_matrix = self.profile_freqs * self.payoff_matrix[..., i]
            fitness_vect = num.safe_divide(
                array1=np.sum(util_matrix, axis=i),
                array2=self.strat_dists[i].strat_freq,
                zero_substitute=0.0)

            num_strats = self.strat_dists[i].num_strats
            fitness_scores[i, 0:num_strats] = fitness_vect

        return fitness_scores
            
    def update_distributions(method="fittest", **kwargs):
        """Assuming updated state, updates internal frequencies of strategies according to provided method. 
           
        Arguments:
            method (str): the rule used to update distributions

            See update_distribution_XXXX family of methods for options.
            If invalid option is presented
        """
        match method.lower():
            case "fittest":
                self.update_distribution_fittest(n)
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
        num_strats = [dist.num_strats for dist in self.strat_dists]
        max_num_strats = self.max_num_strats

        #first, drops the lowest scoring strategies of each country
        for i in range(self.num_countries):

            #extracts information about a given distribution
            dist = self.strat_dists[i]
            scores = self.fitness_scores[i, 0:dist.num_strats]
            
            #initializes information about what to drop    
            left_to_drop = n / 100.0
            drop_amounts = np.zeros_like(scores)

            #orders the strategies by what to drop
            sorted_strats_index = sorted(range(dist.num_strats), key=lambda i: scores[i])

            #drops each relevant amount until reached
            for j in range(num_strats):
                idx = sorted_strat_index[j]
                drop_amounts[idx] = min(left_to_drop, dist.strat_freq[idx])
                left_to_drop -= drop_amounts[idx]

            dist.strat_freq -= drop.amounts

        #then scales the frequencies to reach one
        for i in range(self.num_countries):
            dist.strat_freq /= np.sum(dist.strat_freqs)
