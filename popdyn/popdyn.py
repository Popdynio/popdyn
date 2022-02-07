import math

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

    def __call__(self, groups_pop: dict[str, int]) -> float:
        """
        Applies the differential of the transition over the population data.

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

    def __str__(self) -> str:
        return (
            f'{self.alpha} * {self.beta}' +
            (f' * {" * ".join(self.vars)}' if self.vars else '') +
            (f' / N^{len(self.vars) - 1}' if self.vars and self.N else '')
        )

    def __repr__(self) -> str:
        return self.__str__()


class Model:

    def __init__(self, groups: dict[str, int]) -> None:
        """
        Model that represents the dynamic system of a population. Stores a
        matrix with the transitions between each group.

        Args:
            groups: dictionary that maps the indentifier of each group to the
                value of his population.
        """
        self.groups = groups
        self.matrix: dict[str, dict[str, Transition]] = {g: {} for g in groups}

    def __setitem__(self, start_end: tuple[str], trans: Transition) -> None:
        """
        Adds a transition between to groups to the model.

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
            start_end: tuple containing the the identifiers of start and end
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

    def differential(self, group: str) -> float:
        """
        Applies the equation of transformation for a group based on the
        groups's population.

        Args:
            group: group target of the equation.

        Returns:
            The differential of the group evaluated for the population.
        """
        in_trans = [
            v[group] for v in self.matrix.values()
            if v.get(group) is not None
        ]
        out_trans = [v for v in self.matrix[group].values()]

        return (
            sum([trans(self.groups) for trans in in_trans])
            - sum([trans(self.groups) for trans in out_trans])
        )
