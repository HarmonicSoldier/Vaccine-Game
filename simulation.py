"""This model is responsible for allowing one to construct simulations involving interactions amongst countries according to strategic action.

Simulations are initialized by supplying a list of dictionaries to construct countries from. After constructions, Simulations, following being started, can be progressed, rewound, ran, and reset similar to a VHS tape. During this time, different strategies can be adopted..

Examples:
    >>> country_data1 = {"uid": 1, "name" = "Azkaban"}
    >>> country_data2 = {"uid": 2, "name" = "Bolivia"}
    >>> coop_coeffs = np.array([[1, 0.5], [-0.5, 1]])
    >>> my_simulation = Simulation([country_data1, country_data2], coop_coeffs, 100, 1)
    >>> my_simulation.set_strategy(1, "bang_greed")
    >>> my_simulation.set_strategy(2, "bang_all")
    >>> my_simulation.start_simulation()
    >>> my_simulation.enable_print()
    >>> my_simulation.run_simulation()
    >>> my_simulation.reset_simulation()

    This creates a Simulation between two blank countries called Azkaban and Bolivia.
    The code then runs the simulation, printing the states at each of 100 different time points.
    The code then resets the simulation, erasing all internal record of the simulation.
"""

import numpy as np
import numerics as num
from pandemic import Country, Population, MultiPopulation, Supply, MultiSupply, Preferences, MultiPreferences
from strategy import Strategy

class Simulation:
    """Bundles a family of countries together, connecting them through trade and updating them through simulation logic
    Initializing, starting, and moving through a simulation works similarly to a movie.

    The integer attribute current_time determines the number iterations which have been computed until the present; the range of values for current_time are from 0 to end_time, inclusive. The history of states from 0 until current_time, inclusive, are stored within the attribute state_history. The history of expenditures from 0 until current_time, excluding the right endpoint, are stored in the array expend_history.

    The state of a simulation at any time up to and including the present can be accessed as a numpy array through the method get_state(). The expenditures of the simulation at any point up to yet excluding the present can be accessed as a nupm

    The current state of the simulation can be accessed through the method get_state(), and more general persistent information can be accessed through the method get_traits(). 

    A longer history of the simulation can be accessed through the attributes expend_history and state_history.
    First index indicates
    
    Attributes:
    	current_time (int): the present time being tracked in the simulation
        end_time (int): the final, ending time of the simulation
        time_step (float): the real length of a time step, for scaling purposes
        country_list (:obj:`list` of :obj:`Country`): list of country objects within simulation
        uid_list (:obj:`list` of :obj:`int`): list of uids for the countries being tracked
        num_countries (int): the number of countries present in the simulation
        expend_history (:obj:`np.ndarray` of :obj:`float`): a history of expenditures taken across a simulation until current_time (3D, end_time by num_countries by num_countries)

        	entries for unreached time points are initialized to -1

        state_history (:obj:`np.ndarray` of :obj:`float`): a history of states across a simulation until now (3D, end_time by num_countries by 5)

        	entries for unreached time points are initialized to -1
               
        strategy_list (list Strategy): list of strategy objects for each country
    """

    def __init__(self, country_data, coop_coeffs=None, end_time=10.0, time_step=1):
        """Initializes a simulation from a structure of country data

        Args:
            country_data (:obj:`list` of :obj:`dict`): dictionaries of each country's parameter dictionary of arguments to country constructor
            coop_coeffs (:obj:`np.ndarray` of :obj:`float`, optional): cooperation coefficients for each country with respect to each other country (defaults to identity matrix)
            end_time (:obj:`float`, optional): maximum time of the simulation
            time_step (:obj:`float`, optional): real length of a discrete time step 
        """

        #sets parameters relating to the tracking/progression of time
        self.current_time = None #simulation hasn't started yet
        self.end_time = int(end_time)
        self.time_step = float(time_step)
        self._printing_interval = 1

        #initializes informtation regarding cooperation coefficients beforehand
        if coop_coeffs is None: coop_coeffs = np.eye(len(country_data))

        for i in range(len(country_data)):
            uid_list = [data["uid"] for data in country_data]
            coop_list = coop_coeffs[i, :].tolist()
            country_data[i]["coop_dict"] = dict(zip(uid_list, coop_list))

        #constructs country objects from provided data
        self.country_list = [Country.from_country_data(data) for data in country_data]
        self.uid_list = [country.uid for country in self.country_list]
        self.num_countries = len(self.country_list)


        # #for faster computation, extracts information about relevant parameters
        # self.__cult_rate_array = np.array([supp.cult_rate for supp in self.__supp_list], dtype=float, ndmin=1)

        # self.__max_supply_array = np.array([supp.max_amount for supp in self.__supp_list], dtype=float, ndmin=1)
        # self.__max_consume_array = np.array([supp.max_consume for supp in self.__supp_list], dtype=float, ndmin=1)
        # self.__max_donate_array = np.array([supp.max_donate for supp in self.__supp_list], dtype=float, ndmin=1)
        # self.__max_receive_array = np.array([supp.max_receive for supp in self.__supp_list], dtype=float, ndmin=1)

        #sets default strategy for each country to default list
        self.strategy_list = [Strategy(self.country_list, "default")] * self.num_countries

        #initializes empty arrays ahead of time to store history
        self.expend_history = np.full(
            shape=(end_time, self.num_countries, self.num_countries),
            fill_value=-1.0)
        self.state_history = np.full(
            shape=(end_time + 1, self.num_countries, 5),
            fill_value=-1.0)

    def print_state(self):
        """ Prints the current state of each country."""
        for country in self.country_list:
            country.print_state()

    def print_traits(self):
        """ Prints the persistent traits of each country. """
        for country in self.country_list:
            country.print_traits()

    def set_printing(self, time_interval):
        """Sets time interval for which to print current state of system.

        Args:
            time_interval (int): number of time iterations between automatic state printing
                                 if set to None, disables automatic printing altogether
        """
        self._printing_interval = time_interval

    def set_strategy(self, uid, option="default", kwargs={}):
        """ Sets the strategy for a particular country, based on a provided option

        Args:
            uid (int): (if pos is not specified) the uid of the country
            option (:obj:`str`, optional): used to set strategy to one of a select presets
            kwargs (:obj:`dict`, optional): a dictionary of additional arguments to specify strategy object
        """
        pos = self.uid_list.index(uid)
        self.strategy_list[pos].set_strategy(option, kwargs)

    def set_strategies(self, option_list, kwargs_list):
        """ Sets the strategy for each country all at once, based on provided list of options

        Args:
            option (:obj:`list` of :obj:`str`): used to set strategies to provided presets
            kwargs (:obj:`list` of `dict`): dictionaries of additional arguments to specify strategy
        """
        for i in range(self.num_countries):
            self.set_strategy(self.uid_list[i], option_list[i], kwargs_list[i])

    def replace_strategy(self, uid, strategy):
        """Directly replaces the strategy object of a country.
           Useful in certain applications.

        Args:
            uid (int): (if pos is not specified) the uid of the country
            strat (:obj:`Strategy`): 
        """
        pos = self.uid_list.index(uid)
        self.strategy_list[pos] = strategy

    def get_state(self):
        """ Returns a 2D numpy array representing the current state of each country

        Returns:
            state_array (array float): array of states for each country
            	first dimension: the country with the resources (dimension num_countries)
                second dimension: SIRD, followed by supply count (dimension 5)
        """
        pop_state = self.pop_multi.sird_array.copy()
        sup_state = np.expand_dims(self.supp_multi.amount_array, axis=1)

        state_record = np.append(pop_state, sup_state, axis=1)

        return state_record

    def get_actions(self):
        """
        Returns a 2D numpy array representing the next actions of each country, as determined by the currently set strategies.

        Returns:
            expend_array (array float): (unscaled) array of expenditure actions
        """
        expend_array = np.empty(shape=(self.num_countries, self.num_countries))
        for i in range(self.num_countries):
            strategy = self.strategy_list[i]
            expend_array[i, :] = strategy.get_choice(
                uid=self.uid_list[i],
                expend_history=self.expend_history,
                state_history=self.state_history)

        return expend_array

    def __record_state(self):
        """ Records the current state of each country into the current entry of the history
        """
        self.state_history[self.current_time, :, :] = self.get_state()

    def __record_action(self, expend_record):
        """Records the latest, scaled actions of each country into the action_history variable into the current entry of the history

           Keyword arguments:
           expend_record - a num_countries by num_countries numpy array of expenditures
                           SHOULD BE SCALED BEFOREHAND according to MultiSupply.scale
        """
        self.expend_history[self.current_time, :, :] = expend_record

    def start_simulation(self):
        """ Starts the simulation, allowing for the movement through the simulation
        Bundles populations/preferences/supplies together and sets current time to start
        """
        pop_list = [country.pop for country in self.country_list]
        prefs_list = [country.prefs for country in self.country_list]
        supp_list = [country.supp for country in self.country_list]

        self.pop_multi = MultiPopulation(pop_list)
        self.prefs_multi = MultiPreferences(prefs_list)
        self.supp_multi = MultiSupply(supp_list)

        self.current_time = 0
        self.__record_state()

    def step(self):
        """Executes all neccesary actions to move the simulation forward by a single frame:
           - cultivating/consuming resources according to each country's chosen rate
           - computes and resolves donations, appropriately scaling them to fit limits
           - updates SIRD values in accordance to chosen consumption rates
       """

        #proposes a scheme of expenditures
        expend_array = self.get_actions()

        #scales to meet supply limits
        self.supp_multi.scale_expenditures_all(expend_array, self.pop_multi.sird_array)

        #lastly resolves expenditures
        self.supp_multi.execute_expenditures(expend_array, self.time_step) 

        #next, computes proposed change in the SIRD values for each country, given consumption
        sird_change_array = self.pop_multi.find_cont_change(
            cons_rate_array=np.diagonal(expend_array),
            time=self.time_step,
            steps=num.diffeq_steps)

        #scales the proposed change to fufill positivity conditions
        self.pop_multi.scale_sird_change(sird_change_array)

        #then updates the population by the specified array
        self.pop_multi.execute_sird_change(sird_change_array)

        #then records the actions for the current time period
        self.__record_action(expend_array)

        #then records the state for the next time period while also updating the current time
        self.current_time += 1
        self.__record_state()

        self.pop_multi.to_individual()
        self.supp_multi.to_individual()

        if (self._printing_interval != None):
            if (self.current_time % self._printing_interval == 0):
                self.print_state()

    def progress_simulation(self, time):
        """
        Progresses simulation by the specified time, or to end, whichever comes first

        Args:
            time (int): number of time frames by which to progress the simulation
        """
        for i in range(time):
            print(f"\n Iteration {i} \n")
            self.step()

    def rewind_simulation(self, time_to_rewind):
        """Rewinds simulation by the specified time, or to the beginning
           Erases intermediary frames in the process.

           Args:
               time_to_end (int): number of time frames to rewind / number of iterations to undo
        """

        #if time to rewind is zero, does nothing
        if time_to_rewind == 0:
            return

        #computes the time to return to, with time t=0 being farthest back
        past_time = max(self.current_time - time_to_rewind, 0)

        #updates the state of the system to the past time
        self.pop_multi.sird_array = self.state_history[past_time, :, 0:4].copy()
        self.supp_multi.amount_array = self.state_history[past_time, :, 4].copy()

        #erases the intermediary history by filling with -1
        self.state_history[(past_time + 1):(self.current_time + 1), :, :].fill(-1.0)
        self.expend_history[(past_time):(self.current_time + 1), :, :].fill(-1.0)

        #lastly, sets the current time to specified time
        self.current_time = past_time

    def reset_simulation(self):
        """ Rewinds time of ongoing simulation to the start (t=0).
            Erases all history except for initial state.
        """
        self.rewind_simulation(self.current_time)

    def run_simulation(self):
        """Runs entire simulation from the initial time (t=0) to the end (t=end_time).
           Simulation must be reset manually afterwards.
        """
        self.print_state()
        self.reset_simulation()
        self.progress_simulation(self.end_time)

    def get_payoffs():
        """Returns the payoffs of the current state corresponding to each player

        TODO:
           complete once payoff functions are determined
        """
        return np.zeros(shape=(self.num_countries,))