"""This module provides a family of unit and integration tests to verify the appropriate functioning of the program. Tests here are meant to establish the procedural and the numerical competency of the software. 
Unit tests are based on a shared, singular example case involving two countries with hardcoded results. Integration tests, by contrasts, are ran with either minimal input or randomize input.

Tests are structured according to the pytest format. To run tests here, run pytest on this file.
"""

import copy as copy
import random as rng

import numpy as np
from scipy.integrate import odeint

import numerics as num
from pandemic import (MultiPopulation, MultiPreferences, MultiSupply,
                      Population, Preferences, Resource, Supply)
from simulation import Simulation
from strategy import Strategy, StrategyDistribution
from evolve import EvolveSimulation

from visualize import VisualizeEvolutionSimulation

class GenerateTestCase:
    """Class responsible for generating test cases for reference"""

    test_country_data_0 = {"uid": 0, "name": "Nullland"}

    test_country_data_A = {
        "uid": 1,
        "name": "Azkaban",
        "inf_rate": 0.35,
        "rec_rate": 0.25,
        "dth_rate": 0.02,
        "eff_rate": 0.50,
        "init_sus": 950.0,
        "init_inf": 50.0,
        "sus_util": 1.0,
        "inf_util": 0.5,
        "rec_util": 1.0,
        "dth_util": 0.0,
        "supp_util": 0.2,
        "coop_dict": None,
        "init_supp": 100.0,
        "max_supp": 500.0,
        "max_consume": 50.0,
        "cult_rate": 20.0,
        "max_donate": 40.0,
        "max_receive": 30.0,
        "resource": "TREAT"
    }

    test_country_data_B = {
        "uid": 2,
        "name": "Bolivia",
        "inf_rate": 0.65,
        "rec_rate": 0.35,
        "dth_rate": 0.04,
        "eff_rate": 0.30,
        "init_sus": 900.0,
        "init_inf": 100.0,
        "sus_util": 1.0,
        "inf_util": 0.3,
        "rec_util": 0.8,
        "dth_util": 0.0,
        "supp_util": 0.4,
        "coop_dict": None,
        "init_supp": 150.0,
        "max_supp": 400.0,
        "max_consume": 40.0,
        "cult_rate": 10.0,
        "max_donate": 30.0,
        "max_receive": 30.0,
        "resource": "TREAT"
    }

    @staticmethod
    def gen_country_data_fixed(uid, resource="TREAT"):
        """Copies country data from a hard-coded selection of countries
           Currently supports two countries: 'Noland' (0), 'Azkaban' (1), 'Bolivia' (2).
  
        Args:
           uid (int): uid of the country (if not in provided list, defaults to zero)
           resource (:obj:`str`, optional): type of resource the country is primed to receive
  
        Returns:
            country_data (dict): a dictionary of country attributes for corresponding country
        """
        match uid:
            case 0:
                country_data = copy.deepcopy(GenerateTestCase.test_country_data_0)
            case 1:
                country_data = copy.deepcopy(GenerateTestCase.test_country_data_A)
            case 2:
                country_data = copy.deepcopy(GenerateTestCase.test_country_data_B)
            case _:
                country_data = copy.deepcopy(GenerateTestCase.test_country_data_0)

        country_data["resource"] = resource
        return country_data

    @staticmethod
    def gen_coop_array_fixed(num_countries):
        """Generates a fixed-pattern array of cooperation coefficients.
           Specifically, returns an num_countries by num_countries numpy array with:
           diagonal entries one, upper diagonal entries 0.75, and lower diagonal entries -0.25
  
        Args:
            num_countries (int): the number of countries involved
  
        Returns:
            coop_array (:obj:`np.ndarray` of :obj:`float`): array of cooperation coefficients (2D, num_countries by num_countries)
        """
        diagonal = np.eye(num_countries)
        lower = -0.25 * np.tri(num_countries, k=-1)
        upper = 0.75 * np.tri(num_countries, k=-1).T
        return lower + diagonal + upper

    @staticmethod
    def gen_country_data_random(uid, name, resource="TREAT"):
        """Generates a randomized family of parameters to define a country.
           To ensure uniqueness, uid and name must be manually supplied
  
        Args:
            uid (int): uid of the country whose data is to be generated
            name (str): name of the country whose data is to be generated
            resource (:obj:`str`, optional): type of resource country is to receive
  
        Returns:
            country_data (dict): a dictionary of country attributes for randomized country
        """
        test_country_data_rand = {
            "uid": uid,
            "name": name,
            "inf_rate": 1.0 * rng.random(),
            "rec_rate": 0.25 * rng.random(),
            "dth_rate": 0.1 * rng.random(),
            "eff_rate": rng.random(),
            "init_sus": 1000 * rng.random(),
            "init_inf": 200 * rng.random(),
            "init_sus": 100 * rng.random(),
            "init_inf": 100 * rng.random(),
            "sus_util": rng.random(),
            "inf_util": rng.random(),
            "rec_util": rng.random(),
            "dth_util": rng.random(),
            "supp_util": rng.random(),
            "coop_dict": None,
            "init_supp": 100 * rng.random(),
            "max_supp": 500 * rng.random(),
            "max_consume": 15 * rng.random(),
            "cult_rate": 10 * rng.random(),
            "max_donate": 50 * rng.random(),
            "max_receive": 50 * rng.random(),
            "resource": resource
        }

        return test_country_data_rand

    @staticmethod
    def gen_coop_array_randomized(num_countries):
        """Generates a randomized array of cooperation coefficients.
           Specifically, returns a num_countries by num_countries array with:
           diagonal entries 1, non-diagonal entries uniformly generated between -1 and 1
  
        Args:
            num_countries (int): the number of countries involved (at least one)
  
        Returns:
            coop_array (:obj:`np.ndarray` of :obj:`float`): array of cooperation coefficients (2D, num_countries by num_countries)
        """
        coop_array = 2 * np.random.rand(num_countries, num_countries) - 1
        np.fill_diagonal(coop_array, 1.0)
        return coop_array

class TestPopulationMethods:
    """Tests computations found within the Population/MultiPopulation class for accuracy.
         Uses the shared non-randomized test cases for reference with initial populations
    """
    country_data_A = GenerateTestCase.gen_country_data_fixed(1, "TREAT")
    population_A = Population.from_country_data(country_data_A)

    country_data_B = GenerateTestCase.gen_country_data_fixed(2, "TREAT")
    population_B = Population.from_country_data(country_data_B)

    multi_pop = MultiPopulation([population_A, population_B])

    @staticmethod
    def test_find_sird_deriv():
        """Verifies if find_sird_deriv correctly computes the SIRD equations for initial state"""

        multi_pop = TestPopulationMethods.multi_pop
        
        actual_deriv = np.array([[-16.625, 3.125, 12.5, 1.0],
                                 [-58.5, 19.5, 35.0, 4.0]])

        model_deriv = multi_pop.find_sird_deriv()

        assert np.shape(model_deriv) == (2, 4)
        assert num.is_close(actual_deriv, model_deriv)

    @staticmethod
    def test_cons_deriv():
        """Verifies if cons_deriv correctly computes SIRD equations for initial state"""

        multi_pop = TestPopulationMethods.multi_pop

        resource = Resource("TREAT")
        cons_rate = np.array([20.0, 40.0])
        actual_deriv = np.array([10 * resource.control_vect, 12 * resource.control_vect])
        model_deriv = multi_pop.find_cons_deriv(cons_rate, resource)

        assert np.shape(model_deriv) == (2, 4)
        assert num.is_close(actual_deriv, model_deriv)

    @staticmethod
    def test_scale_sird_change():
        """Verfifies if scale_sird_change correctly scales down excessive sird changes"""

        multi_pop = TestPopulationMethods.multi_pop

        sird_change1 = np.full(shape=(2, 2), fill_value=-200.0)
        sird_change2 = np.full(shape=(2, 2), fill_value=0.0)
        sird_change = np.hstack((sird_change1, sird_change2))

        actual_change = np.array([[-50.0, -50.0, 0, 0], [-100.0, -100.0, 0, 0]])

        multi_pop.scale_sird_change(sird_change)
        model_change = sird_change

        assert np.shape(sird_change) == (2, 4)
        assert num.is_close(actual_change, model_change)

    @staticmethod
    def test_execute_sird_change():
        """Tests for if execute_sird_change properly updates internal sird values"""

        multi_pop = TestPopulationMethods.multi_pop

        sird_change1 = np.full(shape=(2, 2), fill_value=-10.0)
        sird_change2 = np.full(shape=(2, 2), fill_value=0.0)
        sird_change = np.hstack((sird_change1, sird_change2))
        actual_sird = np.array([[940.0, 40.0, 0.0, 0.0], [890, 90.0, 0.0, 0.0]])
        multi_pop.execute_sird_change(sird_change)
        model_sird = multi_pop.sird_array.copy()
        multi_pop.execute_sird_change(-sird_change)

        assert np.shape(multi_pop.sird_array) == (2, 4)
        assert np.array_equal(actual_sird, model_sird)

class TestSupplyMethods():
    """Tests computations found within the Supply/MultiSupply class for accuracy.
       Uses the shared non-randomized test case for reference with initial population.
    """
    country_data_A = GenerateTestCase.gen_country_data_fixed(1, "TREAT")
    supply_A = Supply.from_country_data(country_data_A)

    country_data_B = GenerateTestCase.gen_country_data_fixed(2, "TREAT")
    supply_B = Supply.from_country_data(country_data_B)

    multi_supp = MultiSupply([supply_A, supply_B])

    @staticmethod
    def test_scale_expenditures_max():
        """Verifies that the method scale_expenditures_max appropriately limits expenditures"""
        
        multi_supp = TestSupplyMethods.multi_supp
        
        expend_array = np.array([[100.0, 100.0], [100.0, 100.0]])

        actual_expend = np.array([[50.0, 40.0], [30.0, 40.0]])

        multi_supp.scale_expenditures_max(expend_array)
        model_expend = expend_array

        assert np.shape(model_expend) == (2, 2)
        print(actual_expend)
        print(model_expend)
        assert np.array_equal(actual_expend, model_expend)

    @staticmethod
    def test_scale_collections_max():
        """Verifies that the method scale_collections_max appropriately limits expenditures"""

        multi_supp = TestSupplyMethods.multi_supp

        expend_array = np.array([[100.0, 100.0], [100.0, 100.0]])

        actual_expend = np.array([[50.0, 30.0], [30.0, 40.0]])

        multi_supp.scale_collections_max(expend_array)
        model_expend = expend_array

        assert np.shape(model_expend) == (2, 2)
        print(actual_expend)
        print(model_expend)
        assert np.array_equal(actual_expend, model_expend)

    @staticmethod
    def test_scale_to_supply():
        """Verifies that the method scale_to_supply appropriately scales expenditures to accomodate limited supplies"""

        multi_supp = TestSupplyMethods.multi_supp

        #case 1: one-sided expenditures
        expend_array = np.array([[800.0, 800.0], [0.0, 0.0]])
        actual_expend = np.array([[50.0, 50.0], [0.0, 0.0]])
        multi_supp.scale_to_supply(expend_array)
        model_expend = expend_array
        assert num.is_close(actual_expend, model_expend, buffer=3.0)

        #case 2: consumption only
        expend_array = np.array([[800.0, 0.0], [0.0, 800.0]])
        actual_expend = np.array([[100.0, 0.0], [0.0, 150.0]])
        multi_supp.scale_to_supply(expend_array)
        model_expend = expend_array
        assert num.is_close(actual_expend, model_expend, buffer=3.0)

        #case 3: donation only
        expend_array = np.array([[0.0, 800.0], [800.0, 0.0]])
        actual_expend = np.array([[0.0, 800.0], [800.0, 0.0]])
        multi_supp.scale_to_supply(expend_array)
        model_expend = expend_array
        assert num.is_close(actual_expend, model_expend, buffer=3.0)

        assert np.shape(model_expend) == (2, 2)

    # @staticmethod
    # def test_scale_to_capacity():
    #     """Verifies that the method scale_to_capacity appropriately scales expenditures to accomodate limited storage"""

    #     multi_supp = TestSupplyMethods.multi_supp

    #     #case 1: one-sided expenditures
    #     expend_array = np.array([[800.0, 800.0], [0.0, 0.0]])
    #     actual_expend = np.array([[250.0, 250.0], [0.0, 0.0]])

    #     multi_supp.scale_to_capacity(expend_array)
    #     model_expend = expend_array

    #     #case 2: consumption only
    #     expend_array = np.array([[800.0, 0.0], [0.0, 800.0]])

    #     #case 3: donation only
    #     expend_array = np.array([[0.0, 800.0], [800.0, 0.0]])

    #     assert num.is_close(actual_expend, model_expend)

    @staticmethod
    def test_scale_to_termination():
        """Verifies that the method scale_to_termination appropriately scales an SIRD change to zero for end-epidemic countries"""

        multi_supp = TestSupplyMethods.multi_supp

        sird_array = np.array([[100.0, 100.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])

        expend_array = np.full(shape=(2, 2), fill_value=10.0, dtype=float)
        actual_expend = np.array([[10.0, 10.0], [10.0, 0.0]])

        multi_supp.scale_to_termination(expend_array, sird_array)
        model_expend = expend_array

        assert np.shape(model_expend) == (2, 2)
        print(actual_expend)
        print(model_expend)
        assert np.array_equal(actual_expend, model_expend)


class TestPreferencesMethods:
    """Tests computations found within the Preferences/MultiPreferences class for accuracy.
       Uses the shared non-randomized test case for reference with initial population.
    """

    coop_array = GenerateTestCase.gen_coop_array_fixed(2)
    
    country_data_A = GenerateTestCase.gen_country_data_fixed(1)
    country_data_A["coop_dict"] = {1: coop_array[0, 0], 2: coop_array[0, 1]}
    prefs_A = Preferences.from_country_data(country_data_A)

    country_data_B = GenerateTestCase.gen_country_data_fixed(2)
    country_data_B["coop_dict"] = {1: coop_array[1, 0], 2: coop_array[1, 1]}
    prefs_B = Preferences.from_country_data(country_data_B)

    multi_prefs = MultiPreferences([prefs_A, prefs_B])

    @staticmethod
    def test_get_utility():
        """Verifies that the get_utility method of MultiPopulation against a known result"""

        multi_prefs = TestPreferencesMethods.multi_prefs
        sird_array = np.array([[950.0, 50.0, 0.0, 0.0],
                               [900.0, 100.0, 0.0, 0.0]])
        supp_array = np.array([100.0, 150.0])
        actual_util = np.array([[995, 742.5], [-248.75, 990]])
        model_util = multi_prefs.get_utility(sird_array, supp_array)

        assert np.shape(model_util) == (2, 2)
        assert np.array_equal(actual_util, model_util)

    @staticmethod
    def test_get_utility_net():
        """Verifies that the get_utility_net method of MultiPopulation against a known result"""

        multi_prefs = TestPreferencesMethods.multi_prefs
        sird_array = np.array([[950.0, 50.0, 0.0, 0.0], [900.0, 100.0, 0.0, 0.0]])
        supp_array = np.array([100.0, 150.0])

        actual_util = np.array([1737.5, 741.25])
        model_util = multi_prefs.get_utility_net(sird_array, supp_array)

        assert np.shape(model_util) == (2, )
        assert np.array_equal(actual_util, model_util)


class TestSimulationIndividual:
    """Tests the integrated functionality of the Simulation class with reference to single-person dynamics (i.e. 
      Uses randomized tests for all constitutent tests.
   """

    @staticmethod
    def __check_history(simulation, time):
        """Verifies if the history record is structured as expected"""

        state_history = simulation.state_history
        expend_history = simulation.expend_history

        #checks that the current time equals the expected, provided time
        assert simulation.current_time == time

        #checks that the expend_history array is properly formed
        assert np.all(expend_history[time:, :, :] == -1.0)
        assert num.is_near_positive_array(expend_history[0:time, :, :], buffer=1e-2)

        #checks that the state history array is properly formed
        assert np.all(state_history[(time + 1):, :, :] == -1.0)
        assert num.is_near_positive_array(state_history[0:(time + 1), :, :])

    @staticmethod
    def __check_state(simulation, time):
        """Verifies if the state array is structured as expected"""

        assert np.shape(simulation.pop_multi.sird_array) == (1, 4)
        assert np.array_equiv(simulation.pop_multi.sird_array, simulation.state_history[time, 0, 0:4])
        assert num.is_near_positive_array(simulation.pop_multi.sird_array)

        assert np.shape(simulation.prefs_multi.util_array) == (1, 5)
        assert np.all(np.abs(simulation.prefs_multi.util_array) <= 1.0)
        assert num.is_near_positive_array(simulation.prefs_multi.util_array)

        assert np.shape(simulation.supp_multi.amount_array) == (1, )
        assert np.array_equiv(simulation.supp_multi.amount_array, simulation.state_history[time, 0, 4])
        assert num.is_near_positive_array(simulation.supp_multi.amount_array)

    @staticmethod
    def test_basic_functionality():
        """TEST 1: BASIC FUNCTIONALITY
      Does the basic mechanism of simulation work properly?
      If this fails, there is either a basic programming error or a missing edge case

      Test consists of defining a singular, maximally "blank" country to form a simulation.
      The simulation operations are then executed on the corresponding simulation object.
      """

        end_time = 5
        time_step = 1.0

        country_data = GenerateTestCase.gen_country_data_fixed(0)
        coop_array = np.atleast_2d(np.array([1], dtype=float))

        #initializes the simulation and prints starting states
        print("TASK: Initialize simulation")
        my_simulation = Simulation([country_data], coop_array, end_time,
                                   time_step)
        my_simulation.set_printing(1)
        my_simulation.print_traits()
        print("SUCCESS: Initialize simulation")

        print("TASK: Start simulation")
        my_simulation.start_simulation()
        TestSimulationIndividual.__check_state(my_simulation, 0)
        TestSimulationIndividual.__check_history(my_simulation, 0)
        print("SUCCESS: Initialize simulation")

        print("TASK: Progress simulation")
        my_simulation.progress_simulation(3)
        TestSimulationIndividual.__check_state(my_simulation, 3)
        TestSimulationIndividual.__check_history(my_simulation, 3)
        print("SUCCESS: Progress simulation")

        my_simulation.print_state()

        print("TASK: Run simulation")
        my_simulation.run_simulation()
        TestSimulationIndividual.__check_state(my_simulation, end_time)
        TestSimulationIndividual.__check_history(my_simulation, end_time)
        print("SUCCESS: Run simulation")

        print("TASK: Reset simulation")
        my_simulation.reset_simulation()
        TestSimulationIndividual.__check_state(my_simulation, 0)
        TestSimulationIndividual.__check_history(my_simulation, 0)
        print("SUCCESS: Reset simulation")

    @staticmethod
    def test_sird_updates():
        """TEST 2: SIRD UPDATES
       Do the basic iteration rules for SIRD work properly?
       If this fails first, then there is a miscalculation with SIRD
       """
        end_time = rng.randint(1, 10)
        time_step = 1 * rng.random()

        #Initializes the simulation and verifies it's proper formulation
        country_data = GenerateTestCase.gen_country_data_random(0, "Azkaban")
        coop_array = np.atleast_2d(np.array([1], dtype=float))

        print("TASK: Initialize Simulation")
        my_simulation = Simulation([country_data], coop_array, end_time,
                                   time_step)
        my_simulation.start_simulation()
        my_simulation.set_printing(5)
        TestSimulationIndividual.__check_state(my_simulation, 0)
        TestSimulationIndividual.__check_history(my_simulation, 0)
        print("SUCCESS: Initialize Simulation")

        #Computes the pattern of SIRD over the course of the simulation
        print("TASK: Calculate SIRD Curve")
        my_simulation.run_simulation()
        TestSimulationIndividual.__check_state(my_simulation, end_time)
        TestSimulationIndividual.__check_history(my_simulation, end_time)
        model_sol = my_simulation.get_state()[0, 0:4]
        print("SUCCESS: Calculate SIRD Curve")

        print(f"Model SIRD solution: {model_sol}")

        #Computes the actual, numerical pattern of SIRD over the course of the simulation
        print("TASK: Computing actual solution")
        init_sus = country_data["init_sus"]
        init_inf = country_data["init_inf"]
        inf_rate = country_data["inf_rate"]
        rec_rate = country_data["rec_rate"]
        dth_rate = country_data["dth_rate"]

        init_sird = np.array([init_sus, init_inf, 0, 0])

        def deriv(sird, t):

            component1 = -inf_rate * sird[0] * sird[1] / np.sum(sird)
            component3 = rec_rate * sird[1]
            component4 = dth_rate * sird[1]
            component2 = -component1 - component3 - component4

            return np.array([component1, component2, component3, component4])

        real_sol = odeint(func=deriv,
                          y0=init_sird,
                          t=np.linspace(0, end_time * time_step, 1000))[-1]
        print("SUCCESS: Computing actual solution")

        print(f"Actual SIRD solution: {real_sol}")

        assert num.is_close(model_sol, real_sol, buffer=8.0)

    @staticmethod
    def test_consumption():
        """ TEST 3: CONSUME RESOURCES
       Does consumption have the anticipated effect on the SIRD model?
       If this fails first, there is a miscalculation regarding SIRD and consumption
       """

        end_time = rng.randint(1, 10)
        time_step = 1 * rng.random()

        #Initializes the simulation and verifies it's proper formulation
        country_data = [GenerateTestCase.gen_country_data_random(0, "Azkaban")]
        coop_array = np.atleast_2d(np.array([1], dtype=float))

        #initializes the simulation and prints starting states
        print("TASK: Initialize simulation")
        my_simulation = Simulation(country_data, coop_array, end_time,
                                   time_step)
        my_simulation.start_simulation()
        my_simulation.set_strategy(country_data[0]["uid"], "bang_greed")
        my_simulation.set_printing(None)
        TestSimulationIndividual.__check_state(my_simulation, 0)
        TestSimulationIndividual.__check_history(my_simulation, 0)
        print("SUCCESS: Initialize simulation")

        #Computes the pattern of SIRD over the course of the simulation
        print("TASK: Calculate SIRD Curve")
        my_simulation.run_simulation()
        TestSimulationIndividual.__check_state(my_simulation, end_time)
        TestSimulationIndividual.__check_history(my_simulation, end_time)
        model_sol = my_simulation.get_state()[0, 0:4]
        print("SUCCESS: Calculate SIRD Curve")

        print(f"Model SIRD solution: {model_sol}")

        #Computes the actual, numerical pattern of SIRD over the course of the simulation
        print("TASK: Computing actual solution")
        init_sus = country_data[0]["init_sus"]
        init_inf = country_data[0]["init_inf"]
        inf_rate = country_data[0]["inf_rate"]
        rec_rate = country_data[0]["rec_rate"]
        dth_rate = country_data[0]["dth_rate"]
        eff_rate = country_data[0]["eff_rate"]
        maxi_rate = country_data[0]["max_consume"]
        cult_rate = country_data[0]["cult_rate"]

        init_supp = country_data[0]["init_supp"]
        init_state = np.array([init_sus, init_inf, 0, 0, init_supp])

        def deriv(state, t):
            #computes natural derivative of SIRD process
            sus_deriv = -inf_rate * state[0] * state[1] / np.sum(state[0:5])
            rec_deriv = rec_rate * state[1]
            dth_deriv = dth_rate * state[1]
            inf_deriv = -sus_deriv - rec_deriv - dth_deriv

            if num.is_near_positive(-state[1]):
                cons_rate = 0
            elif (cult_rate < maxi_rate) and num.is_near_positive(-state[4]):
                cons_rate = min(cult_rate, maxi_rate)
            else:
                cons_rate = maxi_rate

            supp_deriv = cult_rate - cons_rate
            rec_deriv += eff_rate * cons_rate
            inf_deriv -= eff_rate * cons_rate

            return np.array(
                [sus_deriv, inf_deriv, rec_deriv, dth_deriv, supp_deriv])

        real_sol = odeint(func=deriv,
                          y0=init_state,
                          t=np.linspace(0, time_step * end_time, 500))[-1]
        print("SUCCESS: Computing actual solution")
        print(f"Actual SIRD solution: {real_sol[0:4]}")

        assert num.is_close(real_sol[0:4], model_sol, buffer=0.05 * np.sum(real_sol[0:4]))

class TestSimulationGroup:
    """Tests the integrated functionality of the Simulation class with reference to multi-person dynamics (i.e. there are multiple individuals)
       Uses randomized tests for all constitutent tests (except basic functionality)
    """

    @staticmethod
    def __check_history(simulation, time):
        """Verifies if the history record is structured as expected"""

        state_history = simulation.state_history
        expend_history = simulation.expend_history

        #checks that the current time equals the expected, provided time
        assert simulation.current_time == time

        #checks that the expend_history array is properly formed
        assert np.all(expend_history[time:, :, :] == -1.0)
        assert num.is_near_positive_array(expend_history[0:time, :, :])

        #checks that the state history array is properly formed
        assert np.all(state_history[(time + 1):, :, :] == -1.0)
        assert num.is_near_positive_array(state_history[0:(time + 1), :, :], buffer=5.0)

    @staticmethod
    def __check_state(simulation, time):
        """Verifies if the state array is structured as expected"""

        assert simulation.current_time == time
        assert np.shape(simulation.pop_multi.sird_array) == (simulation.num_countries, 4)
        assert np.array_equiv(simulation.pop_multi.sird_array, simulation.state_history[time, :, 0:4])
        assert num.is_near_positive_array(simulation.pop_multi.sird_array)
        assert np.shape(simulation.prefs_multi.util_array) == (simulation.num_countries, 5)
        assert np.all(np.abs(simulation.prefs_multi.util_array) <= 1.0)
        assert num.is_near_positive_array(simulation.prefs_multi.util_array)
        assert np.shape(simulation.supp_multi.amount_array) == (simulation.num_countries, )
        assert np.array_equiv(simulation.supp_multi.amount_array, simulation.state_history[time, :, 4])
        assert num.is_near_positive_array(simulation.supp_multi.amount_array)

    @staticmethod
    def test_basic_functionality():
        """TEST 2.1: BASIC FUNCTIONALITY
            Does the basic mechanism of simulation work properly?
            If this fails, there is either a basic programming error or a missing edge case

            Test consists of defining two singular, maximally "blank" countries to form a simulation.
            The simulation operations are then executed on the corresponding simulation object.
        """
        end_time = 5
        time_step = 1.0

        country_data1 = GenerateTestCase.gen_country_data_fixed(0)
        country_data2 = GenerateTestCase.gen_country_data_fixed(0)
        country_data2["uid"] = 1
        country_data2["name"] = "Nullland Two"
        country_data = [country_data1, country_data2]
        coop_array = GenerateTestCase.gen_coop_array_fixed(2)

        #initializes the simulation and prints starting states
        print("TASK: Initialize simulation")
        my_simulation = Simulation(country_data, coop_array, end_time,
                                   time_step)
        my_simulation.set_printing(1)
        my_simulation.print_traits()
        print("SUCCESS: Initialize simulation")

        #computes the pattern of SIRD over the course of the simulation
        print("TASK: Start simulation")
        my_simulation.start_simulation()
        TestSimulationGroup.__check_state(my_simulation, 0)
        TestSimulationGroup.__check_history(my_simulation, 0)
        print("SUCCESS: Start Simulation")

        print("TASK: Progress Simulation")
        my_simulation.progress_simulation(3)
        TestSimulationGroup.__check_state(my_simulation, 3)
        TestSimulationGroup.__check_history(my_simulation, 3)
        print("SUCCESS: Progress Simulation")

        my_simulation.print_state()

        print("TASK: Run simulation")
        my_simulation.run_simulation()
        TestSimulationGroup.__check_state(my_simulation, end_time)
        TestSimulationGroup.__check_history(my_simulation, end_time)
        print("SUCCESS: Run simulation")

        print("TASK: Reset simulation")
        my_simulation.reset_simulation()
        TestSimulationGroup.__check_state(my_simulation, 0)
        TestSimulationGroup.__check_history(my_simulation, 0)
        print("SUCCESS: Reset simulation")

    @staticmethod
    def test_sird_updates():
        """TEST 2: SIRD UPDATES
           Do the basic iteration rules for SIRD work properly?
           If this fails first, then there is a miscalculation with SIRD
        """
        end_time = rng.randint(1, 10)
        time_step = 3 * rng.random()

        #begins by randomly generating up to six countries
        num_countries = rng.randint(2, 6)
        name_list = [
            "Azkaban", "Bolivia", "Centralia", "Ducksburg", "Everyville", "Foxhound"
        ]

        country_data = [
            GenerateTestCase.gen_country_data_random(i, name_list[i], resource="PREVENT")
            for i in range(num_countries)
        ]
        coop_array = GenerateTestCase.gen_coop_array_randomized(num_countries)

        #initializes the simulation and verifies its proper formulation
        print("TASK: Initialize simulation")
        my_simulation = Simulation(country_data, coop_array, end_time,
                                   time_step)
        my_simulation.start_simulation()
        my_simulation.set_printing(5)
        my_simulation.run_simulation()
        TestSimulationGroup.__check_state(my_simulation, end_time)
        TestSimulationGroup.__check_history(my_simulation, end_time)
        print("SUCCESS: Initialize simulation")

        #computes the pattern of SIRD over the course of the simulation
        print("TASK: Calculate SIRD Curve")
        my_simulation.run_simulation()
        TestSimulationGroup.__check_history(my_simulation, end_time)
        TestSimulationGroup.__check_state(my_simulation, end_time)
        model_sol = my_simulation.get_state()[:, 0:4]
        print("SUCCESS: Calculate SIRD Curve")

        print(f"Model SIRD solution: {model_sol}")

        print("TASK: Computing actual solution")

        actual_sol = np.empty(shape=(num_countries, 4), dtype=float)
        for i in range(num_countries):
            #computes the actual SIRD values directly, using python's ODE solver
            init_sus = country_data[i]["init_sus"]
            init_inf = country_data[i]["init_inf"]
            inf_rate = country_data[i]["inf_rate"]
            rec_rate = country_data[i]["rec_rate"]
            dth_rate = country_data[i]["dth_rate"]
            init_sird = np.array([init_sus, init_inf, 0, 0])

            def deriv(sird, t):
                component1 = -inf_rate * sird[0] * sird[1] / np.sum(sird)
                component3 = rec_rate * sird[1]
                component4 = dth_rate * sird[1]
                component2 = -component1 - component3 - component4
                return np.array(
                    [component1, component2, component3, component4])

            actual_sol[i, :] = odeint(func=deriv,
                                      y0=init_sird,
                                      t=np.linspace(0, time_step * end_time, 1000))[-1]

        print("SUCCESS: Computing actual solution")

        assert num.is_close(actual_sol, model_sol, buffer=10)

    @staticmethod
    def test_consumption():
        """TEST 3: resource consumption
            Do the rules for consumption, SIRD, and trade function properly
            If this fails first, then there is a miscaling or error with resources
            (Note: this does not compute a "real" solution, due to emergent complexity
                Rather, it tests for the fufillment of constraints) 
        """
        end_time = rng.randint(1, 10)
        time_step = 1.0 * rng.random()

        #to begin, generates a random pile of country data
        num_countries = rng.randint(2, 6)
        name_list = [
            "Azkaban", "Bolivia", "Centralia", "Ducksburg", "Everyville",
            "Foxhound"
        ]

        country_data = [
            GenerateTestCase.gen_country_data_random(i, name_list[i],
                                                     "PREVENT")
            for i in range(num_countries)
        ]
        coop_array = GenerateTestCase.gen_coop_array_randomized(num_countries)
        np.fill_diagonal(coop_array, 1)

        #Initializes the simulation and prints starting states
        print("TASK: Initialize simulation")
        my_simulation = Simulation(country_data, coop_array, end_time,
                                   time_step)
        my_simulation.start_simulation()
        my_simulation.set_printing(5)
        TestSimulationGroup.__check_state(my_simulation, 0)
        TestSimulationGroup.__check_history(my_simulation, 0)
        print("SUCCESS: Initialize simulation")

        #Uses the model to compute curve of SIRD values
        print("TASK: Calculate SIRD Curve")
        my_simulation.run_simulation()
        model_sol = my_simulation.get_state()
        TestSimulationGroup.__check_state(my_simulation, end_time)
        TestSimulationGroup.__check_history(my_simulation, end_time)
        print("SUCCESS: Calculate SIRD Curve")

        print(f"Model SIRD solution: {model_sol}")

        #does not compute an actual solution due to complexity. Rather validates for structual plausibility
        print("TASK: Validating Solution Bounds")
        assert num.is_near_positive_array(my_simulation.state_history[:, :, :])

        for i in range(num_countries):
            max_consume = country_data[i]["max_consume"]
            assert num.is_near_positive_array(
                max_consume - my_simulation.expend_history[:, i, i])
            max_donate = country_data[i]["max_donate"]
            max_receive = country_data[i]["max_receive"]
            for j in range(num_countries):
                assert (i != j) or num.is_near_positive_array(
                    max_donate - my_simulation.expend_history[:, i, j])
                assert (i != j) or num.is_near_positive_array(
                    max_receive - my_simulation.expend_history[:, j, i])

        print("SUCCESS: Validating Solution Bounds")

class TestEvolveSimulation:
    """Tests the integrated functionality of the EvolveSimulation class to verify desired behaviour"""

    @staticmethod
    def test_basic_functionality():
        """Returns whether or not the evolutionary model runs without issue"""
        country_data1 = GenerateTestCase.gen_country_data_fixed(0, resource="TREAT")
        country_data2 = GenerateTestCase.gen_country_data_fixed(0, resource="TREAT")
        coop_coeffs = np.array([[1, 0.5], [-0.5, 1]])

        my_simulation = Simulation([country_data1, country_data2], coop_coeffs, 100, 1)
        country_list = my_simulation.country_list
        my_simulation.start_simulation()
        my_simulation.set_printing(None)

        strat1A = Strategy(country_list, "idle")
        strat1B = Strategy(country_list, "bang_greed")
        strat_freq = np.array([0.5, 0.5])
        strat_dist1 = StrategyDistribution([strat1A, strat1B], strat_freq)

        strat2A = Strategy(country_list, "idle")
        strat2B = Strategy(country_list, "bang_greed")
        strat_freq = np.array([0.5, 0.5])
        strat_dist2 = StrategyDistribution([strat2A, strat2B], strat_freq)

        contest_time = 10
        my_evolve_simulation = EvolveSimulation(my_simulation, [strat_dist1, strat_dist2], contest_time)
        my_evolve_simulation.run_simulation(method="fittest", n=10.0)
        my_evolve_simulation.reset_simulation()
        my_evolve_simulation.iterate(method="fittest", n=10.0)


    @staticmethod
    def test_evolutionary_updates():
        """Returns whether or not the simulation produces a desired result: strictly dominating strategies always outperform evolutionarily less dominant strategies"""

        #the idea is that, with enough iterations and a very high n, the evolutionary populations should become lobsided almost immediately

        country_data1 = GenerateTestCase.gen_country_data_fixed(1, resource="TREAT")
        country_data2 = GenerateTestCase.gen_country_data_fixed(2, resource="TREAT")
        coop_coeffs = GenerateTestCase.gen_coop_array_fixed(2)

        my_simulation = Simulation([country_data1, country_data2], coop_coeffs, 100, 1)
        country_list = my_simulation.country_list
        my_simulation.start_simulation()
        my_simulation.set_printing(None)

        strat1A = Strategy(country_list, "idle")
        strat1B = Strategy(country_list, "bang_greed")
        strat_freq = np.array([0.5, 0.5])
        strat_dist1 = StrategyDistribution([strat1A, strat1B], strat_freq)

        strat2A = Strategy(country_list, "idle")
        strat2B = Strategy(country_list, "bang_greed")
        strat_freq = np.array([0.5, 0.5])
        strat_dist2 = StrategyDistribution([strat2A, strat2B], strat_freq)

        contest_time = 10
        my_evolve_simulation = EvolveSimulation(my_simulation, [strat_dist1, strat_dist2], contest_time)
        my_evolve_simulation.run_simulation(method="fittest", n=15.0)
        my_evolve_simulation.reset_simulation()
        my_evolve_simulation.iterate(method="fittest", n=15.0)

        visualize_simulation = VisualizeEvolutionSimulation(my_evolve_simulation)
        visualize_simulation.plot_freqs(uid=1)
    