"""This module will be responsible for providing basic visualization tools to view the results of simulations. Tools provided here 
"""
import matplotlib.pyplot as plt
import numpy as np

from simulation import Simulation
from evolve import EvolveSimulation

class VisualizeSimulation:
    """Visualizes the time-series data stored within a simulation

    Attributes:
        simulation (Simulation): the simulation which needs to be visualized
    """

    def __init__(self, simulation):
        self.simulation = simulation

    def plot_sird(self, uid, part_list=('S', 'I', 'R', 'D'), **kwargs):
        """For a given country, plots one or more components (S, I, R, and/or D) as a function of time

        Args:
            uid (int): numeric identifier for country in simulation
            parts(:obj:`list` of :obj:'str'): list of components to plot ("S", "I", "R", and/or "D")
            kwargs (dict): (TBA) additional information to format plotting
        Returns:
            fig (plt.Figure): scaled figure represting the evolution of that particular quantity
        """

        #first computes the relevant data to be plotted
        t_max = self.simulation.current_time
        t_step = self.simulation.time_step
        t_data = np.linspace(0.0, t_max * t_step, num=t_max + 1)
        sird_data = self.simulation.state_history[0:(t_max + 1)].copy()


        #style is not a concern as of now
        fig, ax = plt.subplots(figsize=(3.0, 2.0), layout='constrained')
        ax.annotate('TEST', (1.5, 1.0), ha='center', va='center', color='darkgrey')
        ax.set_xlabel("Time")
        ax.set_ylabel("Population")
        ax.set_title("Population of Country Groups")
        
        if 'S' in part_list:
            ax.plot(t_data, sird_data[:, :, 0], label="Susceptible")
        if 'I' in part_list:
            ax.plot(t_data, sird_data[:, :, 1], label="Infected")
        if 'R' in part_list:
            ax.plot(t_data, sird_data[:, :, 2], label="Recovered")
        if 'D' in part_list:
            ax.plot(t_data, sird_data[:, :, 3], label="Dead")

        ax.legend()

        plt.show()

    def plot_expend(self, uid, uid_list, **kwargs):
        pass

    def plot_util(self, uid):
        pass

class VisualizeEvolutionSimulation:
    """Visualizes the time-series data stored within a simulation

    Attributes:
        simulation (EvolveSimulation): simulation used for the purposes of visualization
    """

    def __init__(self, simulation):
        self.simulation = simulation

    def plot_freqs(self, uid, **kwargs):
        """For a provided list of countries, plots each of their strategy frequency abilities as a function of time

        Args:
            uid (:obj:`int`): uid of country to plot
            **kwargs: additional arguments to be passed into plot construction

        Notes:
            assumes that, over the course of the simulation, strategies are never directly removed
        """
        player_num = self.simulation.simulation.uid_list.index(uid)
        contest_time = self.simulation.current_time

        #initializes a list of strategies to collect information on 
        player_strat_list = self.simulation.history[contest_time][player_num].strat_list
        player_strat_history = np.zeros(shape=(len(player_strat_list), self.simulation.contest_time + 1)) 

        #records the record of strategy frequencies over time
        for t in range(self.simulation.contest_time + 1):
            current_num_strats = self.simulation.history[t][player_num].get_num_strats()
            current_strat_freq = self.simulation.history[t][player_num].strat_freq
            player_strat_history[0:current_num_strats, t] = current_strat_freq

        fig, ax = plt.subplots(figsize=(3.0, 2.0), layout='constrained')
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency")
        ax.set_title("Frequency of Strategies")

        for i in range(len(player_strat_list)):
            freq_history = player_strat_history[i, :]
            t_points = np.arange(start=0, stop=len(freq_history), step=1)
            ax.plot(t_points, freq_history, label=f"{i}")

        ax.legend()
        plt.show()


        
        


    


    



