from popdyn import Model, Transition


# SIR model
groups = {
    's': 10000,
    'i': 500,
    'r': 1000
}

sir = Model(groups)
sir['s', 'i'] = Transition(2, 3, 's', 'i', N=True)
sir['i', 'r'] = Transition(4, 5, 'i')

print(sir)
print('s =', sir.differential('s'))
print('i =', sir.differential('i'))
print('r =', sir.differential('r'))
