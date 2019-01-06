from problem import MultiProblem


class CountingProblem(MultiProblem):
    '''
    A simple problem where actions involve incrementing either of two
    counters to make them sum to 5. This is intentionally designed so
    that
    - there are actions that move away from the goal
    - there are many final states that accomplish the goal
    - an easily-computed heuristic cost is equal to the true optimal cost

    >>> p = CountingProblem(goal_state_sum=5)
    >>> s = p.initial
    >>> for round in range(5):
    ...   assert not p.goal_test(s)
    ...   assert CountingProblem.a_star_heuristic(p, s) == 5 - round
    ...   s = p.result(s, (0, +1))
    >>> assert p.goal_test(s)
    >>> assert CountingProblem.a_star_heuristic(p, s) == 0
    >>> s
    (0, 5)
    '''
    def __init__(self, goal_state_sum=5):
        assert goal_state_sum > 0, 'Goal sum must be positive.'
        initial = (0, 0)
        goals = [
            CountingProblem.predicate_state_sum_at_least(val)
            for val in range(1, goal_state_sum + 1)
        ]
        super(CountingProblem, self).__init__(initial, goals)

    @classmethod
    def a_star_heuristic(cls, problem, state):
        '''
        This heuristic returns a count of incomplete goals.
        '''
        return len(problem.goals) - sum(1 for g in problem.goals if g(state))

    @classmethod
    def predicate_state_sum_at_least(cls, value):
        def pred(state):
            return sum(state) >= value
        return pred

    def actions(self, state):
        # Can increment or decrement either counter.
        return [
            (-1, 0),
            (+1, 0),
            (0, -1),
            (0, +1),
        ]

    def result(self, state, action):
        '''
        This simply applies the changes to each counter to the current state.

        >>> CountingProblem().result((0, 3), (+1, -1))
        (1, 2)
        '''
        return tuple(s + a for s, a in zip(state, action))

    def render(self, state):
        print(state)
