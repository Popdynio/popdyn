import matplotlib.pyplot as plt
from popdyn import Model, Transition


# SIR model
sir_groups = {
    'S': (1-4e-3) * 8.6e4,
    'I': 4e-3 * 8.6e4,
    'R': 0
}
sir = Model(list(sir_groups.keys()))
sir['S', 'I'] = Transition(0.35, 'S', 'I', N=True)
sir['I', 'R'] = Transition(0.035, 'I')

results1 = sir.solve(100, list(sir_groups.values()))
plt.plot(results1['time'], results1['S'], 'r')
plt.plot(results1['time'], results1['I'], 'g')
plt.plot(results1['time'], results1['R'], 'y')
results2 = sir.solve(100, list(sir_groups.values()), solver='TauLeaping')
plt.plot(results2['time'], results2['S'], 'b')
plt.plot(results2['time'], results2['I'], 'm')
plt.plot(results2['time'], results2['R'], 'c')

plt.xlabel('t')
plt.ylabel('groups population')
plt.legend(loc='upper center', ncol=len(sir.groups))
plt.show()
