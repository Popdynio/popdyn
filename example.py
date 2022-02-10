import matplotlib.pyplot as plt

from popdyn import Model, Transition


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
plt.show()
