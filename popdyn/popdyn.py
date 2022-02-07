"""
Module that defines the diferent transitions classes.
"""
import math
import json

class Transition:

    def __init__(
        self,
        alpha: float,
        beta: float,
        *vars: list[str],
        N: bool = False,
    ):
        """
        Class that represents a transition between two groups.

        Args:
            alpha: alpha value of the transition.
            beta: betha balue of the transition.
            vars: different groups identifiers involved in the transition.
            N: True if the transition depend on the global population, False
                in other case.
        """
        self.alpha = alpha
        self.beta = beta
        self.vars = vars
        self.N = N

    def __call__(self, groups_pop: dict[str, int]):
        """
        Aplies the function of transition over the population data.

        Args:
            groups_pop: dictionary that contains for each group indentifier
                the population of that group.
        """
        try:
            vars_pop = [groups_pop[v] for v in self.vars]
        except KeyError as e:
            raise ValueError(f'No population for group "{e.args[0]}"')
        total_pop = sum(p for p in groups_pop.values()) if self.N else 0

        return (
            self.alpha * self.beta * math.prod(vars_pop) /
            pow(total_pop, len(vars_pop) - 1)
        )

    def __str__(self):
        return (
            f'{self.alpha} * {self.beta}' +
            (f' * {" * ".join(self.vars)}' if self.vars else '') +
            (f' / N^{len(self.vars) - 1}' if self.vars and self.N else '')
        )

    def __repr__(self):
        return self.__str__()


class Model:

    def __init__(self, groups: list[str]):
        self.groups: list[str] = groups
        self.matrix: dict[str, dict[str, Transition]] = {g: {} for g in groups}

    def __setitem__(self, start_end: tuple[str], trans: Transition):
        start, end = start_end
        if start not in self.groups:
            raise ValueError('Invalid start group for transition')
        if end not in self.groups:
            raise ValueError('Invalid end group for transition')

        self.matrix[start][end] = trans

    def __getitem__(self, start_end: tuple[str]):
        start, end = start_end
        try:
            return self.matrix[start][end]
        except KeyError:
            return None

    def __str__(self) -> str:
        return '\n'.join([f'{g} -> {self.matrix[g]}' for g in self.matrix])

    def __repr__(self) -> str:
        return self.__str__()

    def differential(self, group: str, groups_pop: dict[str, int]):
        """
        Applies the equation of transformation for a group based on the
        groups's population.

        Args:
            group: group target of the equation.
            groups_pop: dictionary that contains for each group indentifier
                the population of that group.
        """
        in_trans = [
            v[group] for v in self.matrix.values()
            if v.get(group) is not None
        ]
        out_trans = [v for v in self.matrix[group].values()]

        return (
            sum([trans(groups_pop) for trans in in_trans])
            - sum([trans(groups_pop) for trans in out_trans])
        )
