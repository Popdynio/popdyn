import matplotlib.pyplot as plt

from popdyn import Model, Transition


# SIR model
groups = {
    's': 20000,
    'i': 10000,
    'r': 1000
}

sir = Model(groups)
sir['s', 'i'] = Transition(1, 0.0561215, 's', 'i', N=True)
sir['i', 'r'] = Transition(1, 0.0455331, 'i')

# print(sir)
# print('s =', sir.differential('s', groups))
# print('i =', sir.differential('i', groups))
# print('r =', sir.differential('r', groups))

t, pops = sir.solve(100)
print(sir)

for (pop, tag) in zip(pops, sir.groups.keys()):
    plt.plot(t, pop, label=tag)
plt.xlabel('t')
plt.ylabel('groups population')
plt.legend(loc='upper center', ncol=len(sir.groups.keys()))
plt.show()
