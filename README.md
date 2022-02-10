# popdyn

Library for simulation of population dynamics.

Allows to simulate the interaction between specific groups in a population, calculating the amount of members for each group along the time. The transitions between groups are defined using probabilistic rules, and the result is obtained throw the solution of the differential equations involved in the process.

## Installing

```
pip3 install popdyn
```

## Using the library

Let's dive in a simple example to see how it works.

First step is define the model, for this example the [SIR](https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology#The_SIR_model) model.

```python
from popdyn import Model, Transition

# storage in a dict the ID and the initial population for each group
groups = {
    'S': 1000000,
    'I': 10000,
    'R': 5000
}

# intialize the model
sir = Model(list(groups.keys()))

# add transitions between groups
sir['S', 'I'] = Transition(1, 0.0561215, 'S', 'I', N=True)
sir['I', 'R'] = Transition(1, 0.0455331, 'I')
```

If we print the model, we can see the transitions for each group (`N` is the total population):

```
>>> print(sir)
S -> {'I': 1 * 0.0561215 * S * I / N^1}
I -> {'R': 1 * 0.0455331 * I}
R -> {}
```

Once ready the model, we can simulate the behavior of the groups in a period of time:

```python
time, pops = sir.solve(t=100, list(groups.values()))
```

Getting in `time` the points over the time and in `pops` a matrix with the population for each group in the different time points.
