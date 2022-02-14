from typing import Iterable

import numpy as np
from gillespy2 import Model, Parameter, Species, Reaction


class StochasticModel(Model):

    def __init__(self, groups: Iterable[str]):
        Model.__init__(self)
        self.species = {g: Species(name=g, mode='discrete') for g in groups}
        self.add_species(list(self.species.values()))

    def add_transition(
        self,
        src: str,
        dest: str,
        rate: float,
        *involved: tuple[str],
    ) -> None:
        rate = Parameter(expression=rate)
        self.add_parameter(rate)
        reactants = {self.species[g]: 1 for g in involved}
        products = {self.species[g]: 1 for g in involved if g != src}
        products[self.species[dest]] = 2 if dest in involved else 1
        print(f'>>>>>>>>> {reactants=} >>>>>>> {products=}')
        reaction = Reaction(
            name=f'reaction_{src}{dest}',
            rate=rate,
            reactants=reactants,
            products=products,
            # propensity_function=lambda *args: 0.3
        )        
        self.add_reaction(reaction)

    def solve(self, t: int, initial_pops: Iterable[int]):
        self.timespan(np.linspace(0, t, t + 1))
        
        for sp, n in zip(self.species, initial_pops):
            sp.set_initial_value(n)
        
        return self.run(number_of_trajectories=1)[0]