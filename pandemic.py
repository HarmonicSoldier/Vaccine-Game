"""This model is responsible for allowing one to constructinternally coherent Country agents to be placed within the simulation.
Countries possess distinct populations, supplie caches, and preference profiles, including both transient and persistent attributes.
Corresponding attributes/methods are stored into the component classes Population, Supply, and Preferences respectively.

Countries are identified with an integer uid (unique identifier) and a string name, both of which must be specified upon construction.
Other parameters are generally specified through a dictionary of keyword arguments, which can be used to construct components and countries.

Each class here contains a "Multi" variant, which allows for a bundle of those goods to be computed simultaneously.
To avoid redundancy in the API, nearly all computations are exclusive to the Multi variants of objects.
Computations for singlular countries may be obtained by constructing Multi objects from those objects.

Examples:
    This following illustrates the typical way singular objects ought to be constructed.
    >>> my_country_data = {"uid": 1, "name" = "Azkaban", "init_sus = 100.0"}
    >>> my_pop = Population.from_country_data(**my_country_data)
    >>> my_prefs = Preferences.from_country_data(**my_country_data)
    >>> my_supp = Supply.from_country_data(**my_country_data)
    >>> my_country = Country.from_country_data(**data)
    >>> assert my_country = Country(my_country_data["uid"], my_country_data["name"], my_pop, my_prefs, my_supp)

    The following illustrates how computations for singular objects can be performed.
   >>> my_prefs = Preferences.from_country_data(**my_country_data)
   >>> my_multi_prefs = MultiPreferences([my_prefs])
   >>> my_multi_prefs.get_utility(sird_array, supp_array)
"""

import numpy as np
import numerics as num
from scipy.integrate import odeint

class Resource:
    """Defines a type of resource, which is one of the following possible categories:

       "TREAT": resource has a persistent restorative effect, bringing a successfully treated infected individual into full recovery
              associated with a control vector (numpy array) of [0, -1, 1, 0]
       "PREVENT": resource has a persistent preventative effect, preventing a successfully treated susceptible individual from becoming infected
              associated with a control vector (numpy array) of [-1, 0, 1, 0]
    
    Does not identify an amount or efficacy rate of a resource, as these are not intrinsic to the resource itself (efficacy, for example, may vary based on a country's population)

    Attributes:
        type (:obj:`str`): the type of the resource in question
        vect (:obj:`np.ndarray` of :obj:`float`): control vector associated with resource type
    """

    def __init__(self, type):

        self.type = type.upper()
        
        match self.type:
            case "TREAT":
                self.control_vect = np.array([0, -1, 1, 0])
            case "PREVENT":
                self.control_vect = np.array([-1, 0, 1, 0])
            case _:
                self.control_vect = np.array([0, -1, 1, 0])


    def is_terminal(self, sird_array):
        """Given an array of SIRD values for a collection of countries, returns which countries are at a terminal condition, where nonnegativity constraint is active:

        Treatment (TREAT): no infected individuals remaining
        Preventatitive (PREVENT): no susceptible individuals remaining
 
        Args:
             sird_array (:obj:`np.ndarray` of :obj:`float`): array of SIRD values at current time (2D, num_countries by 4)
 
        Returns:
             :obj:`np.ndarray` of :obj:`bool`: True wherever terminal condition is reached (1D, num_countries)
        """
        match self.type:
            case "TREAT":
                return num.which_near_positive(-sird_array[:, 0])
            case "PREVENT":
                return num.which_near_positive(-sird_array[:, 1])

class Country:
    """Represents a self-determining jurisdiction.
    Has own population, resource stockpile, and preferences, stored in attribute objects.
    Attributes and methods relating to a Country's components should be called through constituent attributes.

    Examples:
    >>> my_country = Country.from_country_data
    >>> country_utility = Country.prefs.get_utility(sird_array, supp_array)

    Attributes:
        uid (:obj:`int`): unique identifying number 
        name (:obj:`str`): common name of the country
        pop (:obj:`Population`): stores information about epidemiological populations and parameters 
        prefs (Preferences); stores information about the weights the country associates with different
    	supp (Supply): stores information about a country's current supply of resources and constraints thereof
    """

    def __init__(self, uid, name, pop, prefs, supp):
        self.uid = uid
        self.name = name
        self.pop = pop
        self.prefs = prefs
        self.supp = supp

    @classmethod
    def from_country_data(cls, country_data):
        """Alternative constructor for Country object.
       Given a dictionary of parameters, defines a country with those parameters (see __init__).
       Only required parameters are uid and name.
       Unspecified numeric parameters default to 0.0.

       Args:
           country_data (dict): a dictionary of the following potential arguments:
               uid (:obj:`int`): unique identifying number for the country
               name (:obj:`str`): english name for the country, in any case or format
               inf_rate (:obj:`float`, optional): rate of infection 
               rec_rate (:obj:`float`, optional): rate of recovery 
               dth_rate (:obj:`float`, optional): rate of death
               eff_rate (:obj:`float`, optional): rate of treatment success 
               init_sus (:obj:`float`, optional): the initial number of susceptible individuals
               init_inf (:obj:`float`, optional): the initial number of infected individuals
               init_rec (:obj:`float`, optional): the initial number of recovered individuals
               init_ded (:obj:`float`, optional) the initial number of deceased individuals
               sus_util (:obj:`float`, optional): utility associated with susceptible individuals
               inf_util (:obj:`float`, optional): utility associated with infected individual 
               rec_util (:obj:`float`, optional): utility associated with recovered individual 
               dth_util (:obj:`float`, optional): utility associated with deceased invidual 
               supp_util (:obj:`float`, optional): utility associated with resource
               coop_dict (:obj:`dict`, optional): dictionary of cooperation coefficients
               init_supp (:obj:`float`, optional): the initial quantity of available resources
               max_supp (:obj:`float`, optional): the maximum quantity of resources at a time
               max_consume (:obj:`float`, optional): the maximum rate at which resources can be consumed 
               cult_rate (:obj:`float`, optional): the constant rate at which resources are cultivated 
               max_donate (:obj:`float`, optional): the maximum rate at which the country may donate to any other country 
               max_receive (:obj:`float`, optional): the maximum rate at which country may receive from any other country
               resource (:obj;`str`): the type of resource being utilized by the country
        """
        uid = int(country_data["uid"])
        name = country_data["name"]
        pop = Population.from_country_data(country_data)
        prefs = Preferences.from_country_data(country_data)
        supp = Supply.from_country_data(country_data)

        return Country(uid, name, pop, prefs, supp)

    def print_state(self):
        """ Prints the current state of the country (SIRD and resources)

        Examples:
            >>> example_country.print_state()
            State of country Azkaban (uid 0):
            Susc: 100.0
            Infd: 10.0
            Dead: 2.0
            Supp: 50.0
        """
        print(f"State of country {self.name} (uid {self.uid}) \n")
        print(f" Susc: {self.pop.sird_vect[0]} \n")
        print(f" Infd: {self.pop.sird_vect[1]} \n")
        print(f" Recv: {self.pop.sird_vect[2]} \n")
        print(f" Dead: {self.pop.sird_vect[3]} \n")
        print(f" Supp: {self.supp.amount} \n")

    def print_traits(self):
        """ Prints the persistent traits of the country (parameters)

        Examples:
            >>> example_country.print_traits()
            Paramaters of country Azkaban (uid 0)
            Inf Rate: 0.05
            Rec Rate: 1.25
            Dth Rate: 0.25
            Eff Rate: 0.99

            Resource Type: TREAT
            Max Consumption: 20
            Cultivation: 30
            Max Capacity: 200
            Max Donation: 50
            Max Receive: 150

            Susc Util: 20
            Infd Util: 5
            Recv Util: 20
            Dead Util: 0
            Supp Util: 1

            Coop for uid 1: 0.5
            Coop for uid 2: -0.5
        """

        #population parameters
        print(f"Parameters of country {self.name} (uid {self.uid}) \n")
        print(f" Inf Rate: {self.pop.param_vect[0]} \n")
        print(f" Rec Rate: {self.pop.param_vect[2]} \n")
        print(f" Dth Rate: {self.pop.param_vect[3]} \n")
        print(f" Eff Rate: {self.pop.param_vect[4]} \n")

        #supply parameters
        print(f" Resource Type {self.supp.resource.type} \n")
        print(f" Max Consumption: {self.supp.max_consume} \n")
        print(f" Cultivation: {self.supp.cult_rate} \n")
        print(f" Max Capacity: {self.supp.max_amount} \n")
        print(f" Max Donation: {self.supp.max_donate} \n")
        print(f" Max Receive: {self.supp.max_receive} \n")

        #preference parameters
        print(f" Susc Util: {self.prefs.util_vect[0]} \n")
        print(f" Infd Util: {self.prefs.util_vect[1]} \n")
        print(f" Recv Util: {self.prefs.util_vect[2]} \n")
        print(f" Dead Util: {self.prefs.util_vect[3]} \n")
        print(f" Supp Util: {self.prefs.util_vect[4]} \n")
        for uid in self.prefs._coop_dict.keys():
            print(f"Coop for uid {uid}: {self.prefs._coop_dict[uid]} \n")

class Population:
    """Stores information about an epidemiologically homogenous group of individuals.

    Attributes:
         sird_vect (:obj:`np.ndarray` of :obj:`float`): population counts of each epidemiological compartment (1D, 4)

            Susceptible (S): individuals who have not ever contracted the sickness.
            Infected (I): individiauls who are currently enduring the sickness.
            Recovered (R): individuals who formerly possessed the sickness, yet are alive.
            Dead (D): individuals who formerly possessed the sickness, yet are dead.

            Population numbers can have arbitrary units, but units should be consistent internally and consistent between populations. Further, parameters should utilize time units which are consistent throughout the program.
         param_vect (:obj:`np.ndarray` of :obj:`float`) - epidemiological parameters (1D, 5) of each component (1D, 5): inf_rate, 0, rec_rate, dth_rate, eff_rate
        
       Individuals move from S to I, then from either I to R or I to D based on the following  parameters:

         inf_rate: relative infectivity rate per each pair of present susceptible, infected individuals
         rec_rate: relative recovery rate per each infected individual
         dth_rate: relative death rate per each infected individual
         eff_rate: relative efficacy of treatment resources

       according to the differential equations

        dS/dt = - inf_rate * S * I
        dI/dt = inf_rate*S*I - (rec_rate + dth_rate) * I
        dR/dt = rec_rate * I
        dD/dt = dth_rate * I

        controlled affinely by adding a vector eff_rate(0, -1, 1, 0) or eff_rate(-1, 0, 1, 0) times the rate of consumption to the instaneous rate of change (derivative).

        tot_pop (:obj:`float`): the total population of the country
    """

    def __init__(self, sird_vect, param_vect):

        self.sird_vect = sird_vect.astype(float)
        self.param_vect = param_vect.astype(float)
        self.tot_pop = np.sum(self.sird_vect)

    @classmethod
    def from_country_data(cls, country_data):
        """Constructs a preferences object from a large dictionary of country parameters

        Args:
            country_data (:obj:`list` of :obj:`dict`): dictionaries of each country's parameter dictionary of arguments.

            May include parameters irrelevant to population, but should include all parameters relevant to Population (see Constructor).
        """
        init_sus = country_data.get("init_sus", 0.0)
        init_inf = country_data.get("init_inf", 0.0)
        init_rec = country_data.get("init_rec", 0.0)
        init_ded = country_data.get("init_ded", 0.0)
        sird_vect = np.array([init_sus, init_inf, init_rec, init_ded], dtype=float, ndmin=1)

        inf_rate = country_data.get("inf_rate", 0.0)
        rec_rate = country_data.get("rec_rate", 0.0)
        dth_rate = country_data.get("dth_rate", 0.0)
        eff_rate = country_data.get("eff_rate", 0.0)
        param_vect = np.array([inf_rate, 0, rec_rate, dth_rate, eff_rate], dtype=float, ndmin=1)

        return Population(sird_vect, param_vect)

class MultiPopulation:
    """Stores information about a family of population, allowing for simultaneous computations.

    Attributes:
         sird_array (:obj:`np.ndarray` of :obj:`float`): for each country, population counts of each component (2D, num_countries by 5): susceptible, infected, recovered, and dead

            Individiuals are divided into one of the following components:

                Susceptible (S): individuals who have not ever contracted the sickness.
                Infected (I): individiauls who are currently enduring the sickness.
                Recovered (R): individuals who formerly possessed the sickness, yet are alive.
                Dead (D): individuals who formerly possessed the sickness, yet are dead.

            Population numbers can have arbitrary units, but units should be consistent internally and consistent between populations. Further, parameters should utilize time units which are consistent throughout the program.

         param_array (:obj:`np.ndarray` of :obj:`float`) - for each country, epidemiological parameters (2D, num_countries by 5) of each component (2D, num_countries by 5): inf_rate, 0, rec_rate, dth_rate, eff_rate

       Individuals move from S to I, then from either I to R or I to D based on the following  parameters:

         inf_rate: relative infectivity rate per each pair of present susceptible, infected individuals
         rec_rate: relative recovery rate per each infected individual
         dth_rate: relative death rate per each infected individual
         eff_rate: relative efficacy of treatment resources

       according to the differential equations

        dS/dt = - inf_rate * S * I
        dI/dt = inf_rate*S*I - (rec_rate + dth_rate) * I
        dR/dt = rec_rate * I
        dD/dt = dth_rate * I

        controlled affinely by adding a vector eff_rate(0, -1, 1, 0) or eff_rate(-1, 0, 1, 0) times the rate of consumption to the instaneous rate of change (derivative).

        tot_pop_array (float): for each country, the total population of the country
     """

    def __init__(self, pop_list):
        """ Constructs a MultiPopulation object from an existing list of Population objects

        Args:
            pop_list (:obj:`list` of :obj:`Population`): a list of Population to bundle
        """

        self._pop_list = pop_list
        self._num_countries = len(pop_list)

        self.sird_array = np.array([pop.sird_vect for pop in pop_list], dtype=float, ndmin=2)
        self.param_array = np.array([pop.param_vect for pop in pop_list], dtype=float, ndmin=2)
        self.tot_pop_array = np.array([pop.tot_pop for pop in pop_list], dtype=float, ndmin=1)

    def to_individual(self, pop_list=None):
        """ Stores bundled information into individual Population objects

        Args:
            pop_list (:obj:`list` of :obj:`Population`): a list of Population objects to store information into; By default, this is the list of Populations used to compose bundle
        """
        if pop_list == None: pop_list = self._pop_list
        
        for i in range(self._num_countries):
            self._pop_list[i].sird_vect = self.sird_array[i].copy()

    def find_sird_deriv(self):
        """ Returns the current rate of change of SIRD according to native equations.
            Ignores the effect of control/consumption which is computed independently.

            If any component is non-positive, returns zero vector.

        Returns;
           :obj:`np.ndarray` of :obj:`float`: for each country, rate of change of each SIRD (2D, num_countries by 4)
        """

        #first, computes each of the components directly through the SIRD equations 
        sus_deriv_array = -self.sird_array[:, 0] * self.sird_array[:, 1] * self.param_array[:, 0]
        sus_deriv_array = num.safe_divide(sus_deriv_array, self.tot_pop_array, 0.0)
        rec_deriv_array = self.sird_array[:, 1] * self.param_array[:, 2]
        dth_deriv_array = self.sird_array[:, 1] * self.param_array[:, 3]
        inf_deriv_array = -sus_deriv_array - rec_deriv_array - dth_deriv_array

        #then combines them together in array form
        return np.column_stack((sus_deriv_array, inf_deriv_array, rec_deriv_array, dth_deriv_array))

    def find_cons_deriv(self, cons_rate_array, resource=Resource("TREAT")):
        """Returns current rate of change of SIRD due to affine control.
           Ignores the effect of standard SIRD dynamics, computed independently.

           Returns zero vector for each country with nonpositive infection count.

        Args:
           cons_rate_array (:obj:`np.ndarray` of :obj:`float`): for each country, rate of consumption of relevant resource (1D, num_countries)
           resource (:obj:`Resource`, optional): type of resource being applied

        Returns:
           :obj:`np.ndarray` of :obj:`float`: for each country, control derivative of SIRD
        """
        cons_deriv_array = np.outer(self.param_array[:, 4] * cons_rate_array, resource.control_vect)
        cons_deriv_array *= np.expand_dims(np.logical_not(resource.is_terminal(self.sird_array)), axis=1)

        return cons_deriv_array

    def scale_sird_change(self, change_array):
        """Given a proposed change in SIRD for each country, scales it down to ensure each component of population would nonnegative after resolution

        Args:
             change_array (:obj:`np.ndarray` of :obj:`float`): array of changes in SIRD for each country (2D, num_countries by 4)
        """
        #nonnegative S
        scaling_factors = num.safe_divide(self.sird_array[:, 1], -change_array[:, 1], 1.0)
        np.place(scaling_factors, change_array[:, 1] > 0, 1.0)
        np.place(scaling_factors, scaling_factors > 1, 1.0)

        change_array *=  np.expand_dims(scaling_factors, axis=1)

        #nonnegative I
        scaling_factors = num.safe_divide(self.sird_array[:, 0], -change_array[:, 0], 1.0)
        np.place(scaling_factors, change_array[:, 0] > 0, 1.0)
        np.place(scaling_factors, scaling_factors > 1, 1.0)

        change_array *= np.expand_dims(scaling_factors, axis=1)

        #due to numerical error, change_array may be slightly larger than sird_array
        #corrects for this by replacing change with sird where this occurs
        #this does make the total population technically non-constant, the effect is negligible
        
        where_to_correct = num.which_near_positive(-(self.sird_array + change_array))
        np.copyto(change_array, -self.sird_array, "same_kind", where_to_correct)

        assert num.is_near_positive_array(self.sird_array + change_array)

    def execute_sird_change(self, change_array):
        """Given a proposed change in SIRD population counts for each country, applies it, updating tracked SIRD population counts

        Args:
             change_array (:obj:`np.ndarray` of :obj:`float`): array of changes in SIRD for each country (2D, num_countries by 4)
        """
        self.sird_array += change_array

    def find_cont_change(self, cons_rate_array, resource=Resource("TREAT"), time=1, steps=500):
        """Given a proposed constant rate of consumption for each country, computes the prospective cumulative change in SIRD over a specified duration of time.

        Employs on ODE solver to best approximate continuity of SIRD in intermediate time.
        Result is unscaled and should be scaled appropriately in accordance with needs.

        Args:
            cons_rate_array (:obj:`np.ndarray` of :obj:`float`): scaled rate of consumption at current (1D, num_countries)
            resource (:obj:`Resource`, optional): type of resource being applied
            time (:obj:`int`, optional): amount of time after start to compute new SIRD values
            steps (:obj:`int`, optional): number of intermediary steps to compute with
        """

        #method temporarily updates sird_array during execution, but this is undone at end
        past_sird_array = self.sird_array.copy()

        #as consumption level is constant, pre-computes the (yet unscaled) derivative due to consumption
        cons_deriv_array = self.find_cons_deriv(cons_rate_array, resource)

        #odeint solver requires a linear input, making this extra function a neccesity
        init_sird_linear = self.sird_array.flatten()
        def deriv_net_linear(sird_linear, t):

            #uses the in-built
            self.sird_array = np.reshape(sird_linear, (-1, 4))
            deriv_net_array = self.find_sird_deriv() + cons_deriv_array
            deriv_net_linear = np.ravel(deriv_net_array)
            return deriv_net_linear

        #computes cumulative change over time period using the ODEINT solver
        time_domain = np.linspace(0, time, num=steps)
        change_as_linear = odeint(func=deriv_net_linear, y0=init_sird_linear, t=time_domain)[-1] - init_sird_linear
        change_as_array = np.reshape(change_as_linear, (self._num_countries, 4))
        
        self.sird_array = past_sird_array

        return change_as_array

class Supply:
    """A stockpile of resources, limited by certain constraints.

    Resources can be either applied to one's internal population (consumption) or provided to another country's stockpile (donation). These two actions are called expenditures. Conversely, cultivating resources or receiving a donation from another country is called collection.

    Attributes:
       amount (float): the current supply of available resources (always nonnegative)
       max_amount (float): the maximum capacity of resources that a country may hold at a time
       max_consume (float): the maximum rate of consumption that a country may have at a time
       cult_rate (float): the constant rate which countries may cultivate resources at a time
       max_donate (float): the maximum rate which countries may donate resources to another country
       max_receive (float): the maximum rate which countries may receive resources from another country
       resource (str): the type of resource being utilized by the country

    Supply numbers can have arbitrary units, but units should be consistent internally and consistent between supplies. Further, parameters should utilize time units which are consistent throughout the program
    """

    def __init__(self, amount, max_amount, max_consume, cult_rate, max_donate, max_receive, resource="TREAT"):

        self.amount = float(amount)
        self.max_amount = float(max_amount)
        self.max_consume = float(max_consume)
        self.cult_rate = float(cult_rate)
        self.max_donate = float(max_donate)
        self.max_receive = float(max_receive)
        self.resource = Resource(resource)

    @classmethod
    def from_country_data(cls, country_data):
        """Constructs a supply object from a large dictionary of country parameters.

        Args:
            country_data (:obj:`list` of :obj:`dict`): dictionaries of each country's parameter dictionary of arguments.
        """
        amount = country_data.get("init_supp", 0.0)
        max_amount = country_data.get("max_supp", 0.0)
        max_consume = country_data.get("max_consume", 0.0)
        cult_rate = country_data.get("cult_rate", 0.0)
        max_donate = country_data.get("max_donate", 0.0)
        max_receive = country_data.get("max_receive", 0.0)
        resource = country_data.get("resource", "TREAT")

        return Supply(amount, max_amount, max_consume, cult_rate, max_donate, max_receive, resource)

class MultiSupply:
    """Stores information about a family of supplies, allowing for simultaneous computations
   
       Attributes:
       supp_list (:obj:`list` of :obj:`Suppy`): list of constituent supply objects

       This list is not updated immediately whenever MultiSupply is updated.
       Calling to_individual() updates constituent objects to match

       amount_array (:obj:`np.ndarray` of :obj:`float`): the current supply of available resources (always nonnegative) (1D, num_countries)
       max_amount_array (:obj:`np.ndarray` of :obj:`float`): the maximum capacity of resources that a country may hold at a time (1D, num_countries)
       max_consume_array (:obj:`np.ndarray` of :obj:`float`): the maximum rate of consumption that a country may have at a time (1D, num_countries)
       cult_rate_array (:obj:`np.ndarray` of :obj:`float`): the constant rate which countries may cultivate resources at a time (1D, num_countries)
       max_donate_array (:obj:`np.ndarray` of :obj:`float`): the maximum rate which countries may donate resources to another country  (1D, num_countries)
       max_receive_array (:obj:`np.ndarray` of :obj:`float`): the maximum rate which countries may receive resources from another country (1D, num_countries)
       resource (:obj:`Resource`): the type of resource which the supply carries

    Supply numbers can have arbitrary units, but units should be consistent internally and consistent between supplies. Further, parameters should utilize time units which are consistent throughout the program
    """

    def __init__(self, supp_list):
        """ Class constructor for MultiSupply object. Formed out of individual supply objects.
        Updating the MultiSupply does not update individidual supply

        Args:
            supp_list (:obj:`np.ndarray` of :obj;`Supply`): an existing array of Supply objects to form bundle from
                                                            the resource type should be the same for all 
        """

        #stores references to the list of Supply objects
        self.supp_list = supp_list
        self._num_countries = len(supp_list)

        #extracts attributes for amounts, rates, and maximal limits
        self.amount_array = np.array([supp.amount for supp in supp_list], dtype=float, ndmin=1)
        self.max_amount_array = np.array([supp.max_amount for supp in supp_list], dtype=float, ndmin=1)
        self.max_consume_array = np.array([supp.max_consume for supp in supp_list], dtype=float, ndmin=1)
        self.cult_rate_array = np.array([supp.cult_rate for supp in supp_list], dtype=float, ndmin=1)
        self.max_donate_array = np.array([supp.max_donate for supp in supp_list], dtype=float, ndmin=1)
        self.max_receive_array = np.array([supp.max_receive for supp in supp_list], dtype=float, ndmin=1)

        #combines information about max consumption, max donation into singular array
        self._max_expend_array = np.full(fill_value=self.max_donate_array, shape=(self._num_countries, self._num_countries)).T.copy()
        np.fill_diagonal(self._max_expend_array, val=self.max_consume_array)

        #combines information about max receiving of resources, max consumption into singular array
        self._max_collect_array = np.full(fill_value=self.max_receive_array, shape=(self._num_countries, self._num_countries)).copy()
        np.fill_diagonal(self._max_collect_array, self.max_consume_array)

        #checks if each supply is of the same resource
        self.resource = supp_list[0].resource

    def to_individual(self, supp_list=None):
        """Stores the information contained within bundle into a family of Supply objects

        Args:
            supp_list (:obj:`list` of :obj:`Supply`): list of supply objects to record state to
                                                      defaults to list used to intialized
        Note:
            updates the states of supply_list to match MultiSupply, side effect. No return
        """
        if supp_list is None:
            supp_list = self.supp_list
            for i in range(len(supp_list)):
                supp_list[i].amount = self.amount_array[i]
        else:
            for i in range(len(supp_list)):
                supp_list[i].amount = self.amount_array[i]
                supp_list[i].max_amount = self.max_amount[i]
                supp_list[i].max_consume = self.max_consume[i]
                supp_list[i].cult_rate = self.cult_rate[i]
                supp_list[i].max_donate = self.max_donate[i]
                supp_list[i].max_receive = self.max_receive[i]

    def scale_expenditures_all(self, expend_array, sird_array):
        """Scales a proposed scheme of expenditures to fufill the following requirements:

           - Expenditures/collections fall within hard rate limits for each category
           - After expenditures/collections, each country's supply is between 0 and capacity
           - Consumption is zero for countries which have reached appropriate termination conditions 

        Args:
            expend_array (:obj:`np.ndarray` of :obj:`float`): array of expenditures for each country (2D, num_countries by num_countries)
            sird_array (:obj:`np.ndarray` of :obj:`float`): array of SIRD values at current time (2D, num_countries by 4)
        """
        self.scale_expenditures_max(expend_array)
        self.scale_collections_max(expend_array)
        self.scale_to_supply(expend_array)
        self.scale_to_capacity(expend_array)
        self.scale_to_termination(expend_array, sird_array)
        
    def scale_expenditures_max(self, expend_array):
        """Scales a proposed scheme of expenditures to fit absolute bounds on consumption and donation

        Args:
            expend_array (:obj:`np.ndarray` of :obj:`float`): array of expenditures for each country (2D, num_countries by num_countries)

        Note:
            scales expend_array as a side effect. No return value.
        """
        where_excess = (self._max_expend_array < expend_array)
        np.copyto(expend_array, self._max_expend_array, where=where_excess)

    def scale_collections_max(self, expend_array):
        """Scales a proposed scheme of expenditures to fit absolute bounds on receiving

        Args:
            expend_array (:obj:`np.ndarray` of :obj:`float`): array of expenditures for each country (2D, num_countries by num_countries)

        Note:
            scales expend_array as a side effect. No return value
       """
        where_excess = (self._max_collect_array < expend_array)
        expend_array = np.copyto(expend_array, self._max_collect_array, where=where_excess)

    def scale_to_supply(self, expend_array, time=1):
        """Scales a proposed scheme of expenditures to accomodate limited supply
        
           Scaling is uniform in the sense that, if a plan of expenditures would incur a negative supply at some point, then each component expenditure is scaled down by the same factor until

        Args:
            expend_array (:obj:`np.ndarray` of :obj:`float`): array of expenditures for each country (2D, num_countries by num_countries)
            time (float): amount of time in which resources are to be cultivated/expended

        Note:
            scales expend_array as side effect. No return value
        """

        #scaling down one's country's expenditures may cause other countries to need additoinal scaling
        def needs_scaling():
            change_array = -time * np.sum(expend_array, axis=1)
            buffer = 0.05 * np.average(self.amount_array)
            is_scaled = num.is_near_positive_array(self.amount_array + change_array, buffer)
            return not is_scaled

        iter = 0
        while needs_scaling() and (iter < 5):

            #computes net expenditures for each 
            tot_expend_array = np.sum(time * expend_array, axis=1)
            tot_collect_array = np.sum(time * expend_array, axis=0) - np.diagonal(time * expend_array) + time * self.cult_rate_array
            net_expend_array = tot_expend_array - tot_collect_array

            #scales net expenditures down to attempt to fit supply bounds
            scaling_factors = num.safe_divide(self.amount_array, net_expend_array, 1.0)
            np.place(scaling_factors, net_expend_array <= 0, 1.0)    
            np.place(scaling_factors, scaling_factors > 1, 1.0)
            np.copyto(expend_array, expend_array * np.expand_dims(scaling_factors, axis=1))

            iter += 1

    def scale_to_capacity(self, expend_array, time=1):
        """Scales a proposed scheme of expenditures to accomodate limited capacity of receiving countries

        Args:
            expend_array (:obj:`np.ndarray` of :obj:`float`): array of expenditures for each country (2D, num_countries by num_countries)
            time (float): amount of time in which resources are to be cultivated/expended
        """
        def needs_scaling():
            #allows for moderate overhead in regards to being over the limit
            change_array = np.sum(expend_array, axis=0) - np.diagonal(expend_array) + self.cult_rate_array
            buffer = 0.05 * np.average(self.amount_array)
            is_scaled = num.is_near_positive_array(self.max_amount_array - (self.amount_array + change_array), buffer)
            return not is_scaled

        #computes net expenditures for each country
        iter = 0
        while needs_scaling() and (iter < 5):

            #computes net expenditures for each 
            tot_expend_array = np.sum(expend_array, axis=1)
            tot_collect_array = np.sum(expend_array, axis=0) - np.diagonal(expend_array) + self.cult_rate_array
            net_expend_array = tot_expend_array - tot_collect_array

            #computes the array of quantities of resources left to add to stockpile
            left_to_add_array = self.max_amount_array - self.amount_array

            #scales net expenditures down to attempt to fit supply bounds
            scaling_factors = num.safe_divide(left_to_add_array, -net_expend_array, 1.0)
            np.place(scaling_factors, net_expend_array >= 0, 1.0)    
            np.place(scaling_factors, scaling_factors > 1, 1.0)
            np.copyto(expend_array, expend_array * np.expand_dims(scaling_factors, axis=0))

            iter += 1

    def scale_to_termination(self, expend_array, sird_array):
        """for each country, if epidemic termination condition is reached, scales supply consumption zero

        Args:
            expend_array (:obj:`np.ndarray` of :obj:`float`): array of expenditures for each country (2D, num_countries by num_countries)
            sird_array (:obj:`np.ndarray` of :obj:`float`): array of SIRD values at current time (2D, num_countries by 4)
        """
        is_terminal = self.resource.is_terminal(sird_array)
        new_diag = np.logical_not(is_terminal) * np.diagonal(expend_array)
        np.fill_diagonal(expend_array, new_diag)


    def execute_expenditures(self, expend_array, time_step=1.0):
        """Given an array of expenditures, respectively updates supply amounts 

           To prevent numerical errors, expenditure array SHOULD BE SCALED PRIOR
        Args:
            expend_array (:obj:`np.ndarray` of :obj:`float`): scaled array of expenditure rates for each country (2D, num_countries by num_countries)
            time (:obj:`float`, optional): the duration over which expenditures are to be made
        """
        self.amount_array += time_step * np.sum(expend_array, axis=0) #adds provided donations
        self.amount_array -= time_step * np.sum(expend_array, axis=1) #depletes given donations
        self.amount_array += time_step * self.cult_rate_array         #adds cultivated supply
        self.amount_array -= time_step * np.diagonal(expend_array)    #depletes consumed supply

class Preferences:
    """Stores information about a country's preferences 

    Attributes:
        util_vect (:obj:`np.ndarray` of :obj:`float`): per unit (population unit or supply unit) utilities of the following types (in order):
            sus_util: per unit utility associated with a susceptible individual
            inf_util: per unit utility associated with an infected individual
            rec_util: per unit utility associated with a recovered individual
            dead_util: per unit utility associated with a dead individual
            supp_util: per unit utility associated with a unit of resources

            Each coefficient may be any positive or negative real number. While positive coefficients indicate gain from a particular quantity, negative coefficients indicate loss from a particular quantity, with more positive coefficients indicating stronger value gained from that particular component or lesser loss derived, and more negative coefficients indiciating lesser value gained or greater loss derived.

        coop_vect (:obj:`np.ndarray` of :obj:`float`): array of cooperation coefficients (between -1 and 1) with respect to other countries

            Positive coefficient indicates that one views the other country as an ally; the greater the internal utility of the other country, the greater one's own utility
            Zero coefficient indicates that one views the other country with indifference; the utility of the other country does not affect one's own utility
            Negative coefficients indicate that one views the other country as an enemy; the greater the internal utility of the other country, the lesser one's own utility

            The order of the coefficients is in increasing order of registered uids, stored seperately (see uid_list), 
        
        uid_list (:obj:`list` of :obj:`int`): ordered list of country uid's whose cooperation coefficient is well-defined (including own)
    """

    def __init__(self, util_vect, coop_dict):
        #initializes internal utility coefficients
        assert np.shape(util_vect) == (5,)
        self.util_vect = util_vect.astype(float)

        #initializes external utility coefficients
        self._num_countries = np.size(util_vect)
        self._coop_dict = dict(sorted(coop_dict.items()))
        self.uid_list = list(coop_dict.keys())
        self.coop_vect = np.array(list(self._coop_dict.values()), dtype=float, ndmin=1)

    @classmethod
    def from_country_data(cls, country_data):
        """Constructs a preferences object from a large dictionary of country parameters

        Args:
            country_data (:obj:`list` of :obj:`dict`): dictionaries of each country's parameter dictionary of arguments.
        May include parameters irrelevant to preferences, but should include all parameters relevant to Preferences (utility values, cooperation dictionary):
        """
        sus_util = country_data.get("sus_util", 0.0)
        inf_util = country_data.get("inf_util", 0.0)
        rec_util = country_data.get("rec_util", 0.0)
        dth_util = country_data.get("dth_util", 0.0)
        supp_util = country_data.get("supp_util", 0.0)
        util_vect = np.array([sus_util, inf_util, rec_util, dth_util, supp_util], dtype=float, ndmin=1)
        coop_dict = country_data.get("coop_dict", None)

        return Preferences(util_vect, coop_dict)

class MultiPreferences:
    """A bundle of preferences information for multiple countries, allowing for bunched updates

    Attributes:
        prefs_list(:obj:`list` of :obj:`Preferences`): a list of objects which the MultiPreferences object bundles, with at most one for each country

        util_array (:obj:`np.ndarray` of :obj:`float`): for each country, per-unit utilities (population unit or supply unit) utilities (2D, num_countries by 5):

            sus_util - per unit utility associated with a susceptible individual
            inf_util - per unit utility associated with an infected individual
            rec_util - per unit utility associated with a recovered individual
            dead_util - per unit utility associated with a dead individual
            supp_util - per unit utility associated with a unit of resources

            Each coefficient may be any positive or negative real number. While positive coefficients indicate gain from a particular quantity, negative coefficients indicate loss from a particular quantity, with more positive coefficients indicating stronger value gained from that particular component or lesser loss derived, and more negative coefficients indiciating lesser value gained or greater loss derived.

        coop_array (array float): an array of cooperation coefficients (between -1 and 1) from each country to each other country (2D, num_countries by num_countries):

            Positive coefficient indiciates that a country (given by row number) views another (given by column number) as an ally in regards to calculating utility
            Zero coefficient indiciates that a country (given by row number) views another (given by column number) with indifference in regards to calculating utility
            Negative coefficient indiciates that a country (given by row number) views another (given by column number) as an enemy in regards to calculating utility

            The ordering of coefficients along each row/column is in increasing order of registered uids, stored seperately (see uid_list)

        uid_list (:obj:`list` of :obj:`int`): ordered list of country uid's whose cooperation coefficients are defined within scheme
    """

    def __init__(self, prefs_list):
        """ Initializes a MultiPreferences object out of an existing list of Preferences objects.

       Args:
            prefs_list (list Preferences) - a list of Preferences object to bundle

           Each Preferences object should have the *same* uid_list, which should be of the same length as prefs_list. Further, prefs_list should be ordered in order of increasing uid

        """
        self._num_countries = len(prefs_list)

        #initializes a record of preferences objects; does not possess original references back
        self.prefs_list = prefs_list
        self.util_array = np.array([prefs.util_vect for prefs in prefs_list], dtype=float, ndmin=2)
        self.coop_array = np.array([prefs.coop_vect for prefs in prefs_list], dtype=float, ndmin=2)

    def get_utility(self, sird_array, supp_array):
        """ Given information about the state of each constituent country, returns the components of utility each country enjoys from each other country

        Args:
            sird_array (:obj:`np.ndarray` of 'float'): array storing SIRD values for all countries  (2D, num_countries by 4)
            supp_array (:obj:`np.ndarray` of 'float'): array storing supply amounts for all countries  (1D, num_countries)

        Returns:
            :obj:`np.ndarray` of :obj:`float`: components of utility each country enjoys from each other country (2D, num_countries by num_countries)
        """

        #calculates internal utilities derived from SIRD and supply
        sird_utility_array = np.sum(self.util_array[:, 0:4] * sird_array, axis=1)
        supp_utility_array = self.util_array[:, 4] * supp_array

        #applies the cooperation coefficients to extract pairwise utility
        print(sird_utility_array)
        print(supp_utility_array)
        print(supp_array)
        return self.coop_array * (sird_utility_array + supp_utility_array)

    def get_utility_net(self, sird_array, supp_array):
        """Given information about the state of each constituent country, returns the component of total utility each country derives from the global state
 
        Args:
            sird_array (:obj:`np.ndarray` of 'float'): array storing SIRD values for all countries  (2D, num_countries by 4)
            supp_array (:obj:`np.ndarray` of 'float'): array storing supply amounts for all countries  (1D, num_countries)
 
        Returns:
            :obj:`np.ndarray` of :obj:`float`: total utility of each country (2D, num_countries)
        """
        #calculates internal utilities derived from SIRD and supply
        sird_utility_array = np.sum(self.util_array[:, 0:4] * sird_array, axis=1)
        supp_utility_array = self.util_array[:, 4] * supp_array

        #applies the cooperation coefficeints and sums to extract net utilities
        return np.sum(self.coop_array * (sird_utility_array + supp_utility_array), axis=1)