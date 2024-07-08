#finally runs through the gauntlet of
import numpy as np

from test import GenerateTestCase
from simulation import Simulation
from evolve import EvolveSimulation
from strategy import Strategy, StrategyDistribution
from visualize import VisualizeEvolutionSimulation

country_data = [GenerateTestCase.gen_country_data_random(1, "Candyland")]

country_data1 = GenerateTestCase.gen_country_data_fixed(1, resource="TREAT")
country_data2 = GenerateTestCase.gen_country_data_fixed(2, resource="TREAT")
coop_coeffs = GenerateTestCase.gen_coop_array_fixed(2)

my_simulation = Simulation([country_data1, country_data2], coop_coeffs, 100, 1)
country_list = my_simulation.country_list
my_simulation.start_simulation()
my_simulation.set_printing(None)

strat1A = Strategy(country_list, "bang_all")
strat1B = Strategy(country_list, "bang_greed")
strat_freq = np.array([0.5, 0.5])
strat_dist1 = StrategyDistribution([strat1A, strat1B], strat_freq)

strat2A = Strategy(country_list, "bang_all")
strat2B = Strategy(country_list, "bang_greed")
strat_freq = np.array([0.5, 0.5])
strat_dist2 = StrategyDistribution([strat2A, strat2B], strat_freq)

contest_time = 50
my_evolve_simulation = EvolveSimulation(my_simulation, [strat_dist1, strat_dist2], contest_time)
my_evolve_simulation.run_simulation(method="fittest", n=2.0)
my_evolve_simulation.reset_simulation()

visualize_simulation = VisualizeEvolutionSimulation(my_evolve_simulation)
visualize_simulation.plot_freqs(uid=1)
