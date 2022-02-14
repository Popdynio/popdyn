from __future__ import annotations
import math

import numpy as np
from scipy import integrate
from gillespy2 import Model as GModel, Parameter, Species, Reaction


class Transition:

    def __init__(
        self,
        rate: float,
        *vars: tuple[str],
        N: bool = False,
    ) -> None:
        """
        Class that represents a transition between two groups.

        Args:
            rate: rate change value of the transition.
            vars: different groups identifiers involved in the transition.
            N: True if the transition depends on the global population, False
                in other case.
        """
        if N and not vars:
            raise ValueError(
                'Transition cannot depend on N without interaction of at'
                ' least one group'
            )

        self.rate = rate
        self.vars = vars
        self.N = N

    def __call__(self, vars_pop: list[int], total_pop: int) -> float:
        """
        Applies the differential of the transition over the population data.

        Args:
            vars_pop: population for each group in the transition.
        """
        total_pop = total_pop if self.N else 1

        return (
            self.rate * math.prod(vars_pop) /
            pow(total_pop, len(vars_pop) - 1)
        )

    def __str__(self) -> str:
        return (
            f'{self.rate}' +
            (f' * {" * ".join(self.vars)}' if self.vars else '') +
            (f' / N^{len(self.vars) - 1}' if self.vars and self.N else '')
        )

    def __repr__(self) -> str:
        return self.__str__()


class Model:

    def __init__(self, groups: list[str]) -> None:
        """
        Model that represents the dynamic system of a population. Stores a
        matrix with the transitions between each group.

        Args:
            groups: list with the indentifiers for each group.
        """
        self.groups = groups
        self.matrix: dict[str, dict[str, Transition]] = {g: {} for g in groups}

    def __setitem__(self, start_end: tuple[str], trans: Transition) -> None:
        """
        Adds a transition between two groups to the model.

        Args:
            start_end: tuple containing the the identifiers of start and end
                groups.
        
        Raises:
            ValueError: strart or end group are not registered groups of the
                model.
        """
        start, end = start_end
        if start not in self.groups:
            raise ValueError('Invalid start group for transition')
        if end not in self.groups:
            raise ValueError('Invalid end group for transition')

        self.matrix[start][end] = trans

    def __getitem__(self, start_end: tuple[str]) -> Transition:
        """
        Gets a transition between to groups of the model.

        Args:
            start_end: tuple containing the identifiers of start and end
                groups.
        
        Returns:
            The transition between start and end, None if start and/or end are
            not valid groups.
        """
        start, end = start_end
        try:
            return self.matrix[start][end]
        except KeyError:
            return None

    def __str__(self) -> str:
        return '\n'.join([f'{g} -> {self.matrix[g]}' for g in self.matrix])

    def __repr__(self) -> str:
        return self.__str__()

    def get_in_out_trans(self, group: str) -> tuple[list[Transition], list[Transition]]:
        """
        Gets the incoming and outcoming transitions for a group.

        Args:
            group: target group of transitions.
        
        Returns:
            Tuple containing two lists with the incoming and outcoming
            transitions for the group.
        """
        in_trans = [
            v[group] for v in self.matrix.values()
            if v.get(group) is not None
        ]
        out_trans = [v for v in self.matrix[group].values()]

        return in_trans, out_trans

    def _differential(self, group: str, groups_pop: dict[str, int]) -> float:
        """
        Applies the equation of transformation for a group based on the
        groups's population.

        Args:
            group: group target of the equation.
            groups_pop: population of all the groups of the model for a time t.

        Returns:
            The differential of the group evaluated for the population.
        """
        in_trans, out_trans = self.get_in_out_trans(group)
        total_pop = sum(groups_pop)
        reduced_gs = lambda t: [
            groups_pop[self.groups.index(v)] for v in t.vars]

        return (
            sum([trans(reduced_gs(trans), total_pop) for trans in in_trans])
            - sum([trans(reduced_gs(trans), total_pop) for trans in out_trans])
        )

    def _differential_system(self, groups_pop: list[int], *_) -> tuple[float]:
        """
        Evaluates the differential for each group of the model.

        Args:
            groups_pop: population of all the groups of the model for a time t.
        
        Returns:
            A tuple with the differentials evaluated for each group.
        """
        return tuple(self._differential(g, groups_pop) for g in self.groups)

    def solve(self, t: int, initial_pop: list[int], solver='stochastic') -> dict[str, list[float]]:
        """
        Calculates the evolution of the population for each group over a span
        of time t.

        Args:
            t: total time, it's divided in spans of a unity.
            initial_pop: the initial population values in each group. Must
                keep the order used for the groups when instantiated the model.
            solver: solver to get the results. Options are: 'ode' or 'stochastic'.
        
        Raises:
            ValueError: received an unexpected solver.

        Returns:
            Dictionary containing a key for each grup identifier and a key
            'time', and values for the population for each group in the time
            points.
        """
        if solver == 'stochastic':
            gm = GModel(tspan=np.linspace(0, t, t + 1))
            species = {
                g: Species(name=g, initial_value=v, mode='discrete')
                for g, v in zip(self.groups, initial_pop)
            }
            gm.add_species(list(species.values()))
            for src in self.matrix:
                for dest in self.matrix[src]:
                    t = self.matrix[src][dest]
                    rate = Parameter(
                        name=f'param_{src}{dest}',
                        expression=(t.rate / sum(initial_pop)) if t.N else t.rate
                    )
                    gm.add_parameter(rate)
                    reactants = {species[g]: 1 for g in t.vars}
                    products = {species[g]: 1 for g in t.vars if g != src}
                    products[species[dest]] = 2 if dest in t.vars else 1
                    reaction = Reaction(
                        name=f'reaction_{src}{dest}',
                        rate=rate,
                        reactants=reactants,
                        products=products,
                    )        
                    gm.add_reaction(reaction)
            return gm.run(number_of_trajectories=1)[0]
        elif solver == 'ode':
            time_points = np.arange(t + 1)
            y_result = integrate.odeint(
                func=self._differential_system,
                y0=initial_pop,
                t=time_points,
            )
            return {
                **{'time': time_points},
                **{g: g_values for g, g_values in zip(self.groups, y_result.T)}
            }
        else:
            raise ValueError(
                'Unexpected solver.'
                ' Options available are \'stochastic\' or \'ode\''
            )
