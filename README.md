# popdyn

Library for simulation of population dynamics.

Allows to simulate the interaction between specific groups in a population, calculating the amount of members for each group along the time. The transitions between groups are defined using probabilistic rules. The results can be obtained through one of the deterministic or indeterministic solvers.

## Installing

```
pip3 install popdyn
```

## Using the library

Let's dive in a simple example to see how it works.

First step is define the model, for this example the [SIR](https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology#The_SIR_model) model, whose name comes from the initials S (susceptible), I (infected), R (recovered) and has been quite widespread and used as a basis for more complicated models. In this model, a susceptible can pass to the group of infected if he interacts with an infected and in turn the infected pass to the group of recovered spontaneously. In this model, it is usually taken into account that the probability of contact between a susceptible person and an infected person depends on the number of susceptibles and the proportion of the total population that is made up of infected people. 
 
S(t), I(t), and R(t) represent the number of susceptible, infected, and recovered individuals at time t, respectively. _beta_, known in this case as the transmission rate, represents the number of individuals that go from being susceptible to being infected in a unit of time for each possible effective contact. The number of possible effective contacts is given by the product of S(t) and I(t)/N, which represents the contacts of susceptibles with infected, since the rest of the contacts do not cause a transition from S to I. Note that here it is assumed that contacts occur with equal probability regardless of which group an individual belongs to. In addition to the previous transition, the spontaneous transition from an infected to a recovered one is taken into account, which depends on the number of infected and on _gamma_, called the recovery rate. Finally, the recovery rate is made up of the amount of recoveries that are obtained precisely from the transition from I to R described previously.

The system of differential equations of a deterministic SIR as described above would be:

**dS/dt = - beta *  S(t) * I(t) / N**

**dI/dt = beta * S(t) * I(t) / N - gamma * I(t)**

**dR/dt = gamma * I(t)**

For the example below we consider an initial population composed by 97 susceptible and 3 infected. The values of _beta_ and _gamma_ are 0.35 and 0.035 respectively.

```python
from popdyn import Model, Transition

# storage in a dict the ID and the initial population for each group
groups = {
    'S': 97,
    'I': 3,
    'R': 0
}

# initialize the model
sir = Model(list(groups.keys()))

# add transitions between groups
sir['S', 'I'] = Transition(0.35, 'S', 'I', N=True)
sir['I', 'R'] = Transition(0.035, 'I')
```

If we print the model, we can see the transitions for each group (`N` is the total population):

```
>>> print(sir)
S -> {'I': 0.35 * S * I / N^1}
I -> {'R': 0.035 * I}
R -> {}
```

Once ready the model, we can simulate the behavior of the groups in a period of time using one of the available solvers: `Gillespie`, `TauLeaping` or `ODE`.

```python
results = sir.solve(t=100, initial_pop=list(groups.values()), solver='Gillespie')
```

Getting in `results` a dictionary with a key `'time'` and a key for each group with the population at every time point.
