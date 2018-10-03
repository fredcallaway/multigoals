from queue import Queue
from collections import deque
import itertools
import random


def bfs_search(problem, goal_test=None, root=None, shuffle=True):
    '''Breadth-first search that returns a series of actions to accomplish a goal in a problem.

    Permits custom goal_test and root node, which we use to run tree search from intermediate
    states with changing goal_test. Can select from actions in random order, which can result in
    differences for actions in optimal paths (but not differences in path length!).

    >>> from blockworld import Blockworld
    >>> problem = Blockworld((('C', 'B'), ('A',)), [lambda s: s == (('C', 'B', 'A'), ())])
    >>> bfs_search(problem)
    [('A', 0)]
    '''
    # Borrowing from https://en.wikipedia.org/wiki/Breadth-first_search

    if goal_test is None:
        goal_test = problem.goal_test
    if root is None:
        root = problem.initial

    seen = set()
    q = deque()
    # a dictionary to maintain meta information (used for path formation)
    # key -> (parent state, action to reach child)
    meta = dict()

    meta[root] = (None, None)
    q.append(root)

    # If we're at a goal state, there's no need to take action.
    if goal_test(root):
        return []

    while q:
        s = q.popleft()
        if s in seen:
            continue

        if goal_test(s):
            # If we're at a goal state, we construct the set of actions to achieve this, given our root.
            return construct_path(s, meta)

        seen.add(s)
        actions = problem.actions(s)
        if shuffle:
            random.shuffle(actions)
        for a in actions:
            next_state = problem.result(s, a)
            # HACK we would like to check to see if our state is in q, but we settle for checking against meta
            if next_state not in meta and next_state not in seen:
                q.append(next_state)
                meta[next_state] = (s, a)  # create metadata for these nodes


# Produce a backtrace of the actions taken to find the goal node, using the
# recorded meta dictionary
def construct_path(state, meta):
    action_list = list()

    # Continue until you reach root meta data (i.e. (None, None))
    while meta[state][0] is not None:
        state, action = meta[state]
        action_list.append(action)

    action_list.reverse()
    return action_list


def _all_goal_combinations(problem, state, k):
    '''This is a policy used to select subsets of goals for this problem. This returns all possible subsets
    of the goals.
    '''
    return itertools.combinations(problem.goals, k)


def _all_incomplete_goal_combinations(problem, state, k):
    '''This is a policy used to select subsets of goals for this problem. This returns all possible subsets
    of the goals. However, we do exclude goals that are satisfied given the current state.
    '''
    incomplete = [g for g in problem.goals if not g(state)]
    # We simple return all goals once we have fewer than k left.
    if len(incomplete) < k:
        return [incomplete]
    else:
        return itertools.combinations(incomplete, k)


def _subset_of_ordered_goals(problem, state, k):
    '''This is a policy used to select subsets of goals for this problem. This policy assumes goals are ordered,
    and focused on the k goals that haven't been accomplished yet.

    This function assumes the goals on the problem are ordered, so that an element at index i must be completed
    before all elements at index k > i can be completed. So, we accomplish the goals at the beginning of the list first.
    '''
    # Find first goal that isn't accomplished
    first_not_accomplished = None
    for idx, goal in enumerate(problem.goals):
        if not goal(state):
            first_not_accomplished = idx
            break

    assert first_not_accomplished is not None,\
        'All goals were accomplished for problem {} at state {}'.format(problem, state)

    goals = problem.goals[first_not_accomplished:min(first_not_accomplished + k, len(problem.goals))]
    assert len(goals) == k or first_not_accomplished + k > len(problem.goals), 'Invalid number of goals in subset'
    return [goals]


def solve_using_ordered_goal_subset(problem, k=1, debug=False, action_limit=30):
    '''
    This function returns an action plan, considering a problem with goals that can be considered in order.
    '''
    s = problem.initial
    completed = False
    history = [(None, s)]
    while True and len(history) < action_limit:
        if debug:
            print('State at t={}'.format(len(history)))
            print(problem.render(s))
        if problem.goal_test(s):
            completed = True
            break

        # We find the shortest path that satisfies our subset of goals...
        subsets = _subset_of_ordered_goals(problem, s, k)
        assert len(subsets) == 1, 'Expected 1 goal subset to be returned.'
        subset_goals = subsets[0]
        actions = bfs_search(
            problem,
            root=s,
            goal_test=lambda state: all(g(state) for g in subset_goals))

        # And take first action in this path.
        action = actions[0]
        s = problem.result(s, action)

        history.append((action, s))

    return history, completed


def _considering_a_subset_of_goals(problem, k=1, subset_fn=_all_goal_combinations, debug=False):
    '''
    This function considers a subset of size k of the goals for this problem at each action.

    - `k` - number of goals to consider when choosing an action.
    - `subset_fn` - function used to find subsets of goals.
        Example: _all_goal_combinations returns all goal subsets of size k
    '''
    s = problem.initial
    action = None
    history = []
    while True and len(history) < 20:
        history.append((action, s))
        if debug:
            print('State at t={}'.format(len(history)))
            print(problem.render(s))
        if problem.goal_test(s):
            break
        # For each set of goals, we identify the shortest path.
        action_paths = []
        for subset_goals in subset_fn(problem, s, k):
            action_paths.append(bfs_search(
                problem,
                root=s,
                goal_test=lambda state: all(g(state) for g in subset_goals)))
        # We pick the shortest action path and take one action in that direction
        shortest_path = min(
            action_paths,
            # Path must have non-zero length, as zero length indicates the goal has been satisfied.
            key=lambda actions: float('inf') if len(actions) == 0 else len(actions))
        action = shortest_path[0]
        s = problem.result(s, action)
    num_actions = len(history) - 1
    if problem.goal_test(history[-1][-1]):
        print('Found solution in {} actions'.format(num_actions), [a for (a, s) in history if a])
    else:
        print('Did not find solution in {} actions'.format(num_actions))


if __name__ == '__main__':
    import doctest
    fail_count, test_count = doctest.testmod()
    if not fail_count:
        print('\n\t** All {} tests passed! **\n'.format(test_count))
