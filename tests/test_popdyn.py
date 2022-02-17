import pytest

from popdyn import Model, Transition


class TestTransition:

    t1 = Transition(0.555, 'A')
    t2 = Transition(1/10, 'A', 'B', 'C')
    t3 = Transition(0.123 * 0.444, 'A', 'B', N=True)

    def test___init__(self):
        assert self.t1.vars == ('A',), 'Unexpected vars'
        assert self.t2.vars == ('A', 'B', 'C',), 'Unexpected vars'
        assert self.t3.vars == ('A', 'B',), 'Unexpected vars'

        assert self.t1.N == self.t2.N == False, (
            'Default value of N should be False')
        assert self.t3.N == True, 'N should be True'

        with pytest.raises(ValueError):
            Transition(1.2 * 0.03)

        with pytest.raises(ValueError):
            Transition(0.0872, N=True)

    def test___call__(self):
        assert round(self.t1([100], 100), 10) == 55.5
        assert self.t2([100, 200, 300], 600) == 6e5
        assert self.t3([100, 200], 300) == 3.6408

    def test___str__(self):
        assert str(self.t1) == '0.555 * A'
        assert str(self.t2) == '0.1 * A * B * C'
        assert str(self.t3) == '0.054612 * A * B / N^1'


class TestModel:
    
    def test___init__(self):
        groups = ['A', 'B', 'C', 'D', 'E']
        m = Model(groups)

        assert m.groups == groups

        assert len(m.matrix) == 5
        assert all([g in m.matrix for g in groups])
        assert all([len(m.matrix[g]) == 0 for g in m.matrix])

    def test___setitem__(self):
        m = Model(['A', 'B', 'C'])

        with pytest.raises(ValueError):
            m['A', 'D'] = Transition(1, 'A')
        
        with pytest.raises(ValueError):
            m['E', 'B'] = Transition(1, 'A')
        
        with pytest.raises(ValueError):
            m['E', 'F'] = Transition(1, 'A')

        with pytest.raises(ValueError):
            m['A', 'B'] = Transition(1, 'B', 'C')

        m['A', 'B'] = Transition(1, 'A')
        m['B', 'C'] = Transition(1, 'B')
        m['C', 'A'] = Transition(1, 'C')

        assert all([len(m.matrix[g]) == 1 for g in m.matrix])

    def test___getitem__(self):
        m = Model(['A', 'B', 'C'])
        m['A', 'B'] = Transition(1, 'A')
        m['B', 'C'] = Transition(3, 'B', 'C')
        m['C', 'A'] = Transition(5, 'C', 'A')

        assert m['A', 'B'].rate == 1
        assert m['B', 'C'].rate == 3
        assert m['C', 'A'].rate == 5
        
        assert m['A', 'D'] is None
        assert m['E', 'B'] is None
        assert m['E', 'F'] is None

    def test_get_in_out_trans(self):
        groups = ['A', 'B', 'C', 'D']
        m = Model(groups)
        m['A', 'B'] = Transition(1, 'A')
        m['B', 'C'] = Transition(3, 'B')
        m['D', 'A'] = Transition(7, 'A', 'B', 'C', 'D', N=True)

        a_trans = a_in, a_out = m.get_in_out_trans('A')
        assert all([len(trans) == 1 for trans in a_trans])
        assert (a_in[0].rate, a_out[0].rate) == (7, 1)

        b_trans = b_in, b_out = m.get_in_out_trans('B')
        assert all([len(trans) == 1 for trans in b_trans])
        assert (b_in[0].rate, b_out[0].rate) == (1, 3)

        c_in, c_out = m.get_in_out_trans('C')
        assert len(c_in) == 1
        assert (c_in[0].rate, c_out) == (3, [])

        d_in, d_out = m.get_in_out_trans('D')
        assert len(d_out) == 1
        assert (d_in, d_out[0].rate) == ([], 7)
        
    def test__differential(self):
        groups = ['A', 'B', 'C', 'D']
        pops = [10, 20, 30, 40]
        m = Model(groups)
        m['A', 'B'] = Transition(1 * 2, 'A')
        m['B', 'C'] = Transition(3 * 4, 'B')
        m['C', 'D'] = Transition(5 * 6, 'C', 'D')
        m['D', 'A'] = Transition(7 * 8, 'A', 'B', 'C', 'D', N=True)

        assert m._differential('A', pops) == 13.44 - 20
        assert m._differential('B', pops) == 20 - 240
        assert m._differential('C', pops) == 240 - 36000
        assert m._differential('D', pops) == 36000 - 13.44

    def test_solve(self):
        error = 0.999999
        # returns the dict without the 'time' key
        no_time = lambda d: {k: d[k] for k in d if k != 'time'}

        # SIR Model
        sir_groups = {
            'S': (1-4e-3) * 8.6e4,
            'I': 4e-3 * 8.6e4,
            'R': 0
        }
        sir = Model(list(sir_groups.keys()))
        sir['S', 'I'] = Transition(0.35, 'S', 'I', N=True)
        sir['I', 'R'] = Transition(0.035, 'I')

        sir_ode = sir.solve(100, list(sir_groups.values()), 'ODE')
        sir_ode_last = [sir_ode[g][-1] for g in no_time(sir_ode)]
        assert all([
            abs(sir_ode[g][-1] - v) < error
            for g, v in zip(('S', 'I', 'R', ), [6, 4939, 81053])
        ]), f'SIR ODE solver wrong result: {sir_ode_last}'

        # sir_stoc = sir.solve(100, list(sir_groups.values()))
        # sir_stoc_last = [sir_stoc[g][-1] for g in no_time(sir_stoc)]
        # assert all([
        #     abs(sir_stoc[g][-1] - v) < error
        #     for g, v in zip(('S', 'I', 'R', ), [6, 4939, 81053])
        # ]), f'SIR stochastic solver wrong result: {sir_stoc_last}'

        # SIS model
        sis_groups = {
            'S': 990000,
            'I': 10000,
        }
        sis = Model(list(sis_groups.keys()))
        sis['S','I'] = Transition(4, 'S', 'I', N=True)
        sis['I','S'] = Transition(2, 'I')

        sis_ode = sis.solve(10, list(sis_groups.values()), 'ODE')
        sis_ode_last = [sis_ode[g][-1] for g in no_time(sis_ode)]
        assert all([
            abs(sis_ode[g][-1] - v) < error
            for g, v in zip(('S', 'I', ), [500000, 500000])
        ]), f'SIS ODE solver wrong result: {sis_ode_last}'

        # sis_stoc = sis.solve(100, list(sis_groups.values()))
        # sis_stoc_last = [sis_stoc[g][-1] for g in no_time(sis_stoc)]
        # assert all([
        #     abs(sis_stoc[g][-1] - v) < error
        #     for g, v in zip(('S', 'I', ), [499828, 500712])
        # ]), f'SIS stochastic solver wrong result: {sis_stoc_last}'

        # SEIR model
        seir_groups ={
            'S': 12e6 - 3,
            'E': 0,
            'I': 3,
            'R': 0
        }
        seir = Model(list(seir_groups.keys()))
        seir['S','E'] = Transition(8.4 * 1/7, 'S', 'I', N=True)
        seir['E','I'] = Transition(1/5, 'E')
        seir['I','R'] = Transition(1/7, 'I')

        seir_ode = seir.solve(100, list(seir_groups.values()), 'ODE')
        seir_ode_last = [seir_ode[g][-1] for g in no_time(seir_ode)]
        assert all([
            abs(seir_ode[g][-1] - v) < error
            for g, v in zip(('S', 'E', 'I', 'R', ), [2740, 389, 19208, 11977661])
        ]), f'SEIR ODE solver wrong result: {seir_ode_last}'

        # seir_stoc = seir.solve(1000, list(seir_groups.values()))
        # seir_stoc_last = [seir_stoc[g][-1] for g in no_time(seir_stoc)]
        # assert all([
        #     abs(seir_stoc[g][-1] - v) < error
        #     for g, v in zip(('S', 'E', 'I', 'R', ), [2740, 389, 19208, 11977661])
        # ]), f'SEIR stochastic solver wrong result: {seir_stoc_last}'
