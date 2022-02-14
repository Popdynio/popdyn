from unittest import result
import matplotlib.pyplot as plt
from popdyn import Model, Transition
import gillespy2
import numpy

# SIR model
sir_groups = {
    'S': (1-4e-3) * 8.6e4,
    'I': 4e-3 * 8.6e4,
    'R': 0
}
sir = Model(list(sir_groups.keys()))
sir['S', 'I'] = Transition(1, 0.35, 'S', 'I', N=True)
sir['I', 'R'] = Transition(1, 0.035, 'I')

t, sir_pops = sir.solve(100, list(sir_groups.values()))
print([pop[-1] for pop in sir_pops])
for (pop, tag) in zip(sir_pops, sir.groups):
    plt.plot(t, pop, label=tag)
plt.xlabel('t')
plt.ylabel('groups population')
plt.legend(loc='upper center', ncol=len(sir.groups))
#plt.show()


def a():
    class StochasticSim(gillespy2.Model):
        def __init__(self, parameter_values=None):
            # First call the gillespy2.Model initializer.
            gillespy2.Model.__init__(self, name='Something')

            # Define parameters for the rates of the transitions, Note that the first rate depends on N = 8.6e-4
            betaSI = gillespy2.Parameter(name='betaSI', expression=0.35/8.6e4)
            betaIR = gillespy2.Parameter(name='betaIR', expression=0.035)
            self.add_parameter([betaSI,betaIR])

            # Define variables for the groups representing S, I and R.
            S = gillespy2.Species(name='S', initial_value=(1-4e-3) * 8.6e4)
            I = gillespy2.Species(name='I',   initial_value=4e-3 * 8.6e4)
            R = gillespy2.Species(name='R',   initial_value=0)
            self.add_species([S, I, R])

            # The list of transition sources and results for a Reaction object are each a
            # Python dictionary in which the dictionary keys are Species objects
            # and the values are stoichiometries of the species in the reaction.
            SI = gillespy2.Reaction(name="SI", rate=betaSI, reactants={S:1, I:1}, products={I:2})
            IR = gillespy2.Reaction(name="IR", rate=betaIR, reactants={I:1}, products={R:1})
            self.add_reaction([SI, IR])

            # Set the timespan for the simulation.
            self.timespan(numpy.linspace(0, 100, 101))


    model = StochasticSim()
    results = model.run(number_of_trajectories=1) #Dictionary
    print(results)
    print([x for x in results[0]])
    # plt.plot(results['time'], results['S'], 'r')
    # plt.plot(results['time'], results['I'], 'g')
    # plt.plot(results['time'], results['R'], 'y')
    # plt.show()

from popdyn.stochastic import StochasticModel

sm = StochasticModel(['S', 'I', 'R'])
sm.add_transition('S', 'I', 0.35/8.6e4, 'S', 'I')
sm.add_transition('I', 'R', 0.035, 'I', 'R')
x = sm.solve(100, [(1-4e-3) * 8.6e4, 4e-3 * 8.6e4, 0])