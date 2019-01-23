from collections import deque, defaultdict
import itertools
import random
import heapq


def bfs_search(problem, goal_test=None, root=None, shuffle=True, return_all_equal_cost_paths=False):
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

    if return_all_equal_cost_paths:
        # For each node, the cost of getting from the start node to that node.
        # HACK this is only used to enable the return of equal cost paths.
        gScore = {}
        # The cost of going from start to start is zero.
        gScore[root] = 0
        # Keeping track of the solutions found so far.
        solutions = []

    while q:
        s = q.popleft()
        if s in seen:
            continue

        if goal_test(s):
            if return_all_equal_cost_paths:
                # If we have other solutions, we make sure this one has equal cost.
                if solutions and gScore[solutions[0]] != gScore[s]:
                    # Otherwise, we simply stop looking at solutions. below we return the solutions we found.
                    break
                solutions.append(s)
            else:
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
                if return_all_equal_cost_paths:
                    gScore[next_state] = gScore[s] + 1

    if return_all_equal_cost_paths and solutions:
        return [construct_path(s, meta) for s in solutions]


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


def solve_using_ordered_goal_subset_perf1(problem, k=1, debug=False, action_limit=30, shuffle=True):
    '''
    This function returns an action plan, considering a problem with goals that can be considered in order.

    This variant of solve_using_ordered_goal_subset avoids replanning when possible, preferring to follow
    plans that have already been made. So, in general, this will run bfs_search once for each goal,
    as opposed to what solve_using_ordered_goal_subset does, which is run bfs_search once for each action.
    '''
    s = problem.initial
    completed = False
    history = [(None, s)]
    if debug:
        print('State at t={}'.format(len(history)))
        print(problem.render(s), end='')
    while True and len(history) < action_limit:
        if problem.goal_test(s):
            completed = True
            break

        # We find the shortest path that satisfies our subset of goals...
        subsets = _subset_of_ordered_goals(problem, s, k)
        assert len(subsets) == 1, 'Expected 1 goal subset to be returned.'
        subset_goals = subsets[0]
        next_goal = subset_goals[0]
        assert not next_goal(s), 'We picked a goal that was already satisfied??'

        actions = bfs_search(
            problem,
            root=s,
            shuffle=shuffle,
            goal_test=lambda state: all(g(state) for g in subset_goals))

        for action in actions:
            s = problem.result(s, action)

            history.append((action, s))

            if debug:
                print(f'\nState at t={len(history)}')
                print(problem.render(s), end='')

            # HACK once we've satisfied our next goal, we stop executing this action plan.
            if next_goal(s):
                if debug:
                    print(f'* Halted execution of action plan because we satisfied goal "{next_goal.__name__}".')
                break

    return history, completed


def solve_using_ordered_goal_subset(problem, k=1, debug=False, action_limit=30, shuffle=True):
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
            shuffle=shuffle,
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


def reconstruct_path(cameFrom, current):
    actions = []
    states = [current]
    while current in cameFrom.keys():
        action, current = cameFrom[current]
        actions.append(action)
        states.append(current)
    states.reverse()
    actions.reverse()
    return actions, states


# From https://en.wikipedia.org/wiki/A*_search_algorithm
def A_Star(
    problem,
    heuristic_cost_estimate,
    start=None,
    goal_test=None,
    dist_between=None,
    shuffle=True,
    return_all_equal_cost_paths=False,
    depth_limit=None,
):
    if start is None:
        start = problem.initial
    if goal_test is None:
        goal_test = problem.goal_test
    if dist_between is None:
        dist_between = lambda current, neighbor: 1
    else:
        assert depth_limit is None, (
            'Error: Current depth-limited search is implemented by assuming cost function is 1 so as to use g(state).')

    # The set of nodes already evaluated
    closedSet = set()

    # The set of currently discovered nodes that are not evaluated yet.
    # Initially, only the start node is known.
    openSet = {start}

    # For each node, which node it can most efficiently be reached from.
    # If a node can be reached from many nodes, cameFrom will eventually contain the
    # most efficient previous step.
    cameFrom = {}

    # For each node, the cost of getting from the start node to that node.
    gScore = defaultdict(lambda: float('inf'))  # map with default value of Infinity

    # The cost of going from start to start is zero.
    gScore[start] = 0

    # For each node, the total cost of getting from the start node to the goal
    # by passing by that node. That value is partly known, partly heuristic.
    fScore = defaultdict(lambda: float('inf'))  # map with default value of Infinity

    heap_entry = [0]  # HACK making this an array to make modification easier
    prioritized_nodes = []  # This is a heap

    def set_fscore(node, f):
        fScore[node] = f
        # HACK
        # by adding +heap_entry, we ensure FIFO
        # for LIFO we can add -heap_entry
        # for other kinds of orderings, we can add random.random()
        # TODO figure out why LIFO seems so important for some incidental efficiencies.
        heapq.heappush(prioritized_nodes, (f, -heap_entry[0], node))
        heap_entry[0] += 1

    # For the first node, that value is completely heuristic.
    set_fscore(start, heuristic_cost_estimate(problem, start))

    if return_all_equal_cost_paths:
        solutions = []

    while openSet:
        f, _, current = heapq.heappop(prioritized_nodes)
        if current in closedSet:
            # HACK this can happen when we have set the f-score for a state
            # twice (thus adding it to queue) before we process it. Since we
            # always process things in order of f-score, we can assume a
            # second visit to this node can be ignored as it's about a path
            # that is less optimal.
            continue
        if goal_test(current):
            if return_all_equal_cost_paths:
                # If we have other solutions, we make sure this one has equal cost.
                if solutions and gScore[solutions[0]] != gScore[current]:
                    # Otherwise, we simply stop looking at solutions. below we return the solutions we found.
                    break
                solutions.append(current)
            else:
                return reconstruct_path(cameFrom, current)

        openSet.remove(current)
        closedSet.add(current)

        # Depth limit is the number of nodes deep we will search. We skip nodes any deeper than limit.
        # So for limit of 3, we will visit nodes 3 actions in from the root.
        if depth_limit is not None and gScore[current] >= depth_limit:
            assert gScore[current] == depth_limit, 'A* depth-limited should not descend past depth limit.'
            if return_all_equal_cost_paths:
                # Then we start gathering with logic akin to the above for equal cost paths.
                # If we have other solutions, we make sure this one has equal fScore.
                # We compare fScore since that's how we prioritize nodes and since our
                # only sensible way to assess distance to the goal involves our heuristic
                # of the cost.
                # HACK determine if we simply prefer to compare the heuristic cost instead of fScore.
                # HACK gScore above is really equal to fScore as h() is 0. so this is possibly equivalent logic??
                if solutions and fScore[solutions[0]] != fScore[current]:
                    # Otherwise, we simply stop looking at solutions. below we return the solutions we found.
                    break
                solutions.append(current)
                continue  # We want to ensure we avoid searching any deeper.
            else:
                return reconstruct_path(cameFrom, current)

        actions = problem.actions(current)
        if shuffle:
            random.shuffle(actions)
        for a in actions:
            neighbor = problem.result(current, a)
            if neighbor in closedSet:
                continue  # Ignore the neighbor which is already evaluated.

            # The distance from start to a neighbor
            tentative_gScore = gScore[current] + dist_between(current, neighbor)

            if neighbor not in openSet:  # Discover a new node
                openSet.add(neighbor)
            elif tentative_gScore >= gScore[neighbor]:
                continue  # This is not a better path.

            # This path is the best until now. Record it!
            cameFrom[neighbor] = (a, current)
            gScore[neighbor] = tentative_gScore
            set_fscore(neighbor, gScore[neighbor] + heuristic_cost_estimate(problem, neighbor))

    if return_all_equal_cost_paths and solutions:
        return [reconstruct_path(cameFrom, s) for s in solutions]


def _count_goals_accomplished_in_order(state, goals):
    '''
    We count the number of goals that have been accomplished. We intentionally only count
    the goals in order as all other goals that are currently satisfied will later have
    to be unsatisfied to complete the problem. So, the number of all satisfied goals
    is equal to or larger than the number returned by this function since we exclude goals
    that are not accomplished in proper order.
    In the case of the Blockworld domain, this counts the number of blocks that are in
    the stack we are building up.

    >>> goals = [lambda x: x > 1, lambda x: x < 4, lambda x: x % 2 == 0]
    >>> s=2
    >>> _count_goals_accomplished_in_order(s, goals), sum(1 for g in goals if g(s))
    (3, 3)
    >>> s=3
    >>> _count_goals_accomplished_in_order(s, goals), sum(1 for g in goals if g(s))
    (2, 2)
    >>> s=4
    >>> _count_goals_accomplished_in_order(s, goals), sum(1 for g in goals if g(s))
    (1, 2)
    '''
    accomplished_count = 0
    for g in goals:
        if g(state):
            accomplished_count += 1
        else:
            break
    return accomplished_count


def make_ordered_k_goal_cost_heuristic(problem, start, k, include_accomplished=True, debug=False):
    '''
    Given a problem and start state and limit on goal computation k, we generate a heuristic
    for A* search and next goals to consider that starts from that start state and has limited subset of
    goals at most size k to consider.

    Can optionally ensure that previously accomplished goals will be included in heuristic counts
    and list of next goals. This avoids undoing work that has already been completed. This isn't
    always ideal, but depending on your goal structure, may be more optimal.
    '''

    # Find # of goals that are accomplished from our current start.
    accomplished_count = _count_goals_accomplished_in_order(start, problem.goals)

    # Find the list of k (or less) goals that have not been accomplished yet.
    next_goals = problem.goals[:min(accomplished_count+k, len(problem.goals))]
    next_goal = next_goals[accomplished_count]

    if debug:
        print('Goals to solve:', [g.__name__ for g in next_goals[accomplished_count:]])

    if not include_accomplished:
        next_goals = next_goals[accomplished_count:]
        assert len(next_goals) == k or accomplished_count + len(next_goals) == len(problem.goals)

    def heuristic_cost(_, state):
        # Return the # of our ~k goals that are not accomplished. We ensure they're counted
        # in order.
        k_accomplished = _count_goals_accomplished_in_order(state, next_goals)
        return len(next_goals) - k_accomplished

    def goal_test(state):
        return all(g(state) for g in next_goals)

    return next_goal, goal_test, heuristic_cost


def solve_using_ordered_goal_subset_astar(problem, k=1, debug=False, action_limit=30, shuffle=True):
    st = problem.initial
    history = [(None, st)]
    completed = False

    while True and len(history) < action_limit:
        if debug:
            print(f'State at t={len(history)}')
            print(problem.render(st), end='')
        if problem.goal_test(st):
            completed = True
            break
        next_goal, goal_test, pred = make_ordered_k_goal_cost_heuristic(problem, st, k, debug=debug)
        actions, states = A_Star(
            problem, pred,
            start=st,
            goal_test=goal_test,
            shuffle=shuffle)
        if debug:
            print(f'Planned to get to {states[-1]} via actions {actions}')
        for a, s in zip(actions, states[1:]):
            history.append((a, s))
            if debug:
                print(f'\nState at t={len(history)}')
                print(problem.render(s), end='')
            if next_goal(s):
                if debug:
                    print(f'* Halted execution of action plan because we satisfied goal "{next_goal.__name__}".')
                break
        # Start from the state our search left us at.
        st = history[-1][-1]

    if debug:
        print('actions', len(history)-1, [a for a, s in history if a])

    return history, completed


def compute_action_path_probabilities(paths, _current_step=0):
    '''
    This function takes a set of action paths and produces the probability of
    each path assuming an agent took a random action at each choice point.

    Returns dictionary with path (as tuple) mapping to probablity of path.
    '''
    if _current_step == 0:
        assert len(paths), 'Paths were not passed in.'
        # Make paths into tuples so we can have simple dict indexing
        paths = [tuple(p) for p in paths]
        # Assert the paths are distinct
        assert len(set(paths)) == len(paths), 'Found identical paths.'
        # Assert the paths are of the same length
        assert all(len(p) == len(paths[0]) for p in paths), 'Found paths with mismatched length.'

    # If we have one path, we always choose that path.
    if len(paths) == 1:
        return {paths[0]: 1}

    # Otherwise, we group by different steps we can take at this current step.
    grouped = {}
    for path in paths:
        grouped.setdefault(path[_current_step], []).append(path)

    # We assume random selection among the possibilities.
    random_p = 1./len(grouped)

    result = {}
    for start, same_start in grouped.items():
        same_start_p = compute_action_path_probabilities(
            same_start, _current_step=_current_step + 1)
        # Probability of these actions at future steps is combined
        # with probability of taking this step's action.
        for path, p in same_start_p.items():
            result[path] = random_p * p

    return result


if __name__ == '__main__':
    import doctest
    fail_count, test_count = doctest.testmod()
    if not fail_count:
        print('\n\t** All {} tests passed! **\n'.format(test_count))
