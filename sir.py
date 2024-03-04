# I'm hoping to use pdoc3 to generate automatic documenation
import numpy as np
import math as mt

# # def si_deriv([SI], infection_gain_rate, infection_loss_rate)
# """ Computes derivatives for S, I in SIR model  """
#     S0, I0 = SI
#     infection_gain = beta*S*I
#     infection_loss = -(recovery_rate + death_rate)*I
#     dS = -infection_gain
#     dI = inection_gain - infection_loss
#
#     return [dS, dI]

# infection, recovery, and death rates are assumed constant over the interval

def sird_iterate(SIRD, infection_rate, recovery_rate, death_rate, time_step):

    SI = np.array([SIRD[0], SIRD[1]])
    RD = np.array([SIRD[2], SIRD[3]])
    # we need only calculate new S, I values using RK4,

    half_time_step = 0.5 * time_step
    infection_loss_rate = recovery_rate + death_rate

    # uses information to compute differentials
    def si_change(SI):
        Sc = SI[0]
        Ic = SI[1]
        gain = infection_rate*Sc*Ic
        return np.array([-gain, gain - infection_loss_rate * Ic])

    k1 = si_change(SI)
    k2 = si_change(SI + half_time_step * k1)
    k3 = si_change(SI + half_time_step * k2)
    k4 = si_change(SI + time_step * k3)

    dSI = (time_step/6)*(k1 + 2*(k2 + k3) + k4)
    dRD = -np.sum(dSI) * (np.array([recovery_rate, death_rate]))/(recovery_rate + death_rate)

    SIRD = np.concatenate((SI + dSI, RD + dRD), axis=0)

    return SIRD


# Total time
total_time = 10

# Time step interval
time_step = 0.05

# SIRS epidemic parameters
infection_rate = 0.15
recovery_rate = 0.5
death_rate = 0.01

# Initial distribution of S, I, R, D
SIRD = np.array([95, 5, 0, 0])

for t in range(mt.floor(total_time/time_step)):
    SIRD = sird_iterate(SIRD, infection_rate, recovery_rate, death_rate, time_step)
    print(SIRD, np.sum(SIRD))
