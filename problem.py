class Problem(object):
    """The abstract class for a formal problem.  

    Adapted from https://github.com/aimacode/aima-python

    You should subclass this and implement the methods actions and result, and
    possibly __init__, goal_test, and path_cost. Then you will create
    instances of your subclass and solve them with the various search
    functions.
    """

    def __init__(self, initial=None, goal=None):
        """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal.  Your subclass's constructor can add
        other arguments."""
        self.initial = initial
        self.goal = goal

    def actions(self, state):
        """Return the actions that can be executed in the given
        state. The result would typically be a list, but if there are
        many actions, consider yielding them one at a time in an
        iterator, rather than building them all at once."""
        raise NotImplementedError

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        raise NotImplementedError

    def goal_test(self, state):
        """Return True if the state is a goal. The default method compares the
        state to self.goal or checks for state in self.goal if it is a
        list, as specified in the constructor. Override this method if
        checking against a single self.goal is not enough."""
        try:
            return (state in self.goal) or (state == self.goal)
        except:
            return state == self.goal

class MultiProblem(Problem):
    """A problem specified by multiple simultaneous goals.

    Goals is a sequence of functions of type state -> bool."""
    def __init__(self, initial=None, goals=None):
        self.initial = initial
        self.goals = goals
    
    def goal_test(self, state):
        return all(goal(state) for goal in self.goals)
