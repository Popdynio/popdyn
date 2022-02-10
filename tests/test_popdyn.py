from black import Mode
from numpy import matrix
import pytest

from popdyn import Model, Transition


class TestTransition:

    t0 = Transition(1.2, 0.03)
    t1 = Transition(1, 0.555, 'A')
    t2 = Transition(1, 1/10, 'A', 'B', 'C')
    t3 = Transition(0.123, 0.444, 'A', 'B', N=True)

    def test___init__(self):
        assert self.t0.vars == (), 'Expected empty vars'
        assert self.t1.vars == ('A',), 'Unexpected vars'
        assert self.t2.vars == ('A', 'B', 'C',), 'Unexpected vars'
        assert self.t3.vars == ('A', 'B',), 'Unexpected vars'

        assert self.t0.N == self.t1.N == self.t2.N == False, (
            'Default value of N should be False')
        assert self.t3.N == True, 'N should be True'

        with pytest.raises(ValueError):
            Transition(0.0872, 2, N=True)

    def test___call__(self):
        assert self.t0([], 100) == 0.036
        assert round(self.t1([100], 100), 10) == 55.5
        assert self.t2([100, 200, 300], 600) == 6e5
        assert self.t3([100, 200], 300) == 3.6408

    def test___str__(self):
        assert str(self.t0) == '1.2 * 0.03'
        assert str(self.t1) == '1 * 0.555 * A'
        assert str(self.t2) == '1 * 0.1 * A * B * C'
        assert str(self.t3) == '0.123 * 0.444 * A * B / N^1'


class TestModel:
    
    def test___init__(self):
        m = Model(['A', 'B', 'C', 'D', 'E'])
        assert m.groups == ['A', 'B', 'C', 'D', 'E']
        assert len(m.matrix) == 5

    def test___set_item__(self):
        m = Model(['A', 'B', 'C'])

        with pytest.raises(ValueError):
            m['A', 'D'] = Transition(1, 1)
        
        with pytest.raises(ValueError):
            m['E', 'B'] = Transition(1, 1)
        
        with pytest.raises(ValueError):
            m['E', 'F'] = Transition(1, 1)

    def test___getitem__(self):
        m = Model(['A', 'B', 'C'])
        m['A', 'B'] = Transition(1, 2)
        m['B', 'C'] = Transition(3, 4)
        m['C', 'A'] = Transition(5, 6)

        assert m['A', 'B'].alpha == 1
        assert m['B', 'C'].alpha == 3
        assert m['C', 'A'].alpha == 5
        
        assert m['A', 'D'] is None
        assert m['E', 'B'] is None
        assert m['E', 'F'] is None

    def test__differential(self):
        groups = ['A', 'B', 'C', 'D']
        pops = [10, 20, 30, 40]
        m = Model(groups)
        m['A', 'B'] = Transition(1, 2)
        m['B', 'C'] = Transition(3, 4, 'B')
        m['C', 'D'] = Transition(5, 6, 'C', 'D')
        m['D', 'A'] = Transition(7, 8, 'A', 'B', 'C', 'D', N=True)

        assert m._differential('A', pops) == 13.44 - 2
        assert m._differential('B', pops) == 2 - 240
        assert m._differential('C', pops) == 240 - 36000
        assert m._differential('D', pops) == 36000 - 13.44

    def test_solve(self):
        error = 0.999999
        # SIR Model
        sir_groups = {
            'S': (1-4e-3) * 8.6e4,
            'I': 4e-3 * 8.6e4,
            'R': 0
        }
        sir = Model(list(sir_groups.keys()))
        sir['S', 'I'] = Transition(1, 0.35, 'S', 'I', N=True)
        sir['I', 'R'] = Transition(1, 0.035, 'I')

        _, sir_pops = sir.solve(100, list(sir_groups.values()))
        assert all([
            abs(pop[-1] - v) < error
            for pop, v in zip(sir_pops, [7, 5115, 80877])
        ])

        # SIS model
        sis_groups = {
            'S': 990,
            'I': 10,
        }
        sis = Model(list(sis_groups.keys()))
        sis['S','I'] = Transition(4, 1, 'S', 'I', N=True)
        sis['I','S'] = Transition(1, 2, 'I')

        _, sis_pops = sis.solve(10, list(sis_groups.values()))
        assert all([
            abs(pop[-1] - v)
            for pop, v in zip(sis_pops, [500, 500])
        ])

        # SEIR model
        seir_groups ={
            'S': 12e6 - 3,
            'E': 0,
            'I': 3,
            'R': 0
        }
        seir = Model(list(seir_groups.keys()))
        seir['S','E'] = Transition(1, 8.4 * 1/7, 'S', 'I', N=True)
        seir['E','I'] = Transition(1, 1/5, 'E')
        seir['I','R'] = Transition(1, 1/7, 'I')

        _, seir_pops = seir.solve(100, list(seir_groups.values()))
        assert all([
            abs(pop[-1] - v)
            for pop, v in zip(seir_pops, [2746, 469, 22066, 11974717])
        ])