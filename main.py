#finally runs through the gauntlet of
from test import GenerateTestCase

from simulation import Simulation
from visualize import VisualizeSimulation

country_data = [GenerateTestCase.gen_country_data_random(1, "Candyland")]
end_time = 1000
time_step = 0.1

my_simulation = Simulation(country_data, end_time=end_time, time_step=time_step)
my_simulation.set_strategy(1, "bang_greed")

my_simulation.start_simulation()
my_simulation.run_simulation()

view_simulation = VisualizeSimulation(my_simulation)
view_simulation.plot_sird(1)
