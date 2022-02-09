import math

import numpy as np
from scipy import integrate


class Transition:

    def __init__(
        self,
        alpha: float,
        beta: float,
        *vars: list[str],
        N: bool = False,
    ) -> None:
        """
        Class that represents a transition between two groups.

        Args:
            alpha: alpha value of the transition.
            beta: beta balue of the transition.
            vars: different groups identifiers involved in the transition.
            N: True if the transition depends on the global population, False
                in other case.
        """
        self.alpha = alpha
        self.beta = beta
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
            self.alpha * self.beta * math.prod(vars_pop) /
            pow(total_pop, len(vars_pop) - 1)
        )

    def __str__(self) -> str:
        return (
            f'{self.alpha} * {self.beta}' +
            (f' * {" * ".join(self.vars)}' if self.vars else '') +
            (f' / N^{len(self.vars) - 1}' if self.vars and self.N else '')
        )

    def __repr__(self) -> str:
        return self.__str__()


class Model:

    def __init__(self, initial_groups: dict[str, int]) -> None:
        """
        Model that represents the dynamic system of a population. Stores a
        matrix with the transitions between each group.

        Args:
            groups: dictionary that maps the indentifier of each group to the
                initial value of his population.
        """
        self.groups = initial_groups
        self.matrix: dict[str, dict[str, Transition]] = {
            g: {} for g in initial_groups}

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

    def _differential(self, group: str, groups_pop: dict[str, int]) -> float:
        """
        Applies the equation of transformation for a group based on the
        groups's population.

        Args:
            group: group target of the equation.
            groups_pop: groups_pop: population of all the groups of the model
                for a time t.

        Returns:
            The differential of the group evaluated for the population.
        """
        in_trans = [
            v[group] for v in self.matrix.values()
            if v.get(group) is not None
        ]
        out_trans = [v for v in self.matrix[group].values()]

        total_pop = sum(self.groups.values())
        gs_keys = list(self.groups.keys())
        reduced_gs = lambda t: [groups_pop[gs_keys.index(v)] for v in t.vars]

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
            A tuple wiht the differentials evaluated for each group.
        """
        return tuple(self._differential(g, groups_pop) for g in self.groups)

    def solve(self, t: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates the evolution of the population for each group over a span
        of time t.

        Args:
            t: total time, it's divided in timespans of a unity.
        
        Returns:
            Tuple conatinig an array of points over the timespan and a matrix
            with the value of population for each group in the time points.
        """
        time_points = np.arange(t)
        y_result = integrate.odeint(
            func=self._differential_system,
            y0=list(self.groups.values()),
            t=time_points,
        )

        return time_points, y_result.T