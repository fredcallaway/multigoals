from simple_spatial import SimpleSpatial
from blockworld import Blockworld
import agent

# This task is solved suboptimally unless you consider all 3 goals. So, we have a pretty coarse
# test of correctness in this file by making sure our k=2 solvers are suboptimal with a 9-move
# solution.
m = [
    "...C.",
    "B..B.",
    ".....",
    "A..A.",
    ".S...",
]
spatial_task_1 = SimpleSpatial(((1, 4), ()), m, [
    SimpleSpatial.make_visited_goal(symbol) for symbol in ['A', 'B', 'C']])
spatial_task_actions = [
    None, (-1, 0), (0, -1), (1, 0), (1, 0), (1, 0), (0, -1), (0, -1), (0, -1)]


def test_solve_using_ordered_goal_subset():
    history, completed = agent.solve_using_ordered_goal_subset(spatial_task_1, k=2, shuffle=False)
    assert completed
    actions = [a for a, s in history]
    assert len(actions) == 9
    assert actions == spatial_task_actions


def test_solve_using_ordered_goal_subset_perf1():
    history, completed = agent.solve_using_ordered_goal_subset_perf1(spatial_task_1, k=2, shuffle=False)
    assert completed
    actions = [a for a, s in history]
    assert len(actions) == 9
    assert actions == spatial_task_actions


def test_solve_using_ordered_goal_subset_astar():
    # Strangely, first two actions switched with a*
    spatial_task_actions = [
        None, (0, -1), (-1, 0), (1, 0), (1, 0), (1, 0), (0, -1), (0, -1), (0, -1)]
    history, completed = agent.solve_using_ordered_goal_subset_astar(spatial_task_1, k=2, shuffle=False)
    assert completed
    actions = [a for a, s in history]
    assert len(actions) == 9
    assert actions == spatial_task_actions


def test_make_ordered_k_goal_cost_heuristic():
    def make_state_contains_pred(item):
        return lambda state: item in state

    class MockProblem(object):
        def __init__(self, goals):
            self.goals = goals

    goals = [make_state_contains_pred(i) for i in range(10)]
    p = MockProblem(goals)

    next_goal, goal_test, h = agent.make_ordered_k_goal_cost_heuristic(p, [0], 3, include_accomplished=False)
    assert h(p, [0]) == 3
    assert h(p, [0, 1]) == 2
    assert h(p, [0, 1, 2]) == 1
    assert h(p, [0, 1, 2, 3]) == 0
    assert goal_test([0, 1, 2, 3])
    # HACK goal test works here because we don't need to include accomplished
    assert goal_test([1, 2, 3])

    next_goal, goal_test, h = agent.make_ordered_k_goal_cost_heuristic(p, range(8), 3, include_accomplished=False)
    assert h(p, range(8)) == 2
    assert h(p, range(9)) == 1
    assert h(p, range(10)) == 0
    assert goal_test(range(10))
    # HACK goal test works here because we don't need to include accomplished
    assert goal_test(range(7, 10))

    next_goal, goal_test, h = agent.make_ordered_k_goal_cost_heuristic(p, [0], 3, include_accomplished=True)
    assert h(p, [0]) == 3
    assert h(p, [0, 1]) == 2
    assert h(p, [0, 1, 2]) == 1
    assert h(p, [0, 1, 2, 3]) == 0
    assert goal_test([0, 1, 2, 3])
    assert not goal_test([1, 2, 3])

    next_goal, goal_test, h = agent.make_ordered_k_goal_cost_heuristic(p, range(8), 3, include_accomplished=True)
    assert h(p, range(8)) == 2
    assert h(p, range(9)) == 1
    assert h(p, range(10)) == 0
    assert goal_test(range(10))
    assert not goal_test(range(7, 10))


def test_astar_return_all_equal_cost_paths():
    problem = Blockworld(
        (('D', 'A'), ('C', 'B'), ()),
        [
            Blockworld.make_above_predicate(top, bottom)
            for (top, bottom) in [('C', 'D'), ('B', 'C'), ('A', 'B')]
        ],
    )
    next_goal, goal_test, h = agent.make_ordered_k_goal_cost_heuristic(
        problem, problem.initial, k=1, debug=True)
    solutions = agent.A_Star(
        problem,
        h,
        start=problem.initial,
        goal_test=goal_test,
        return_all_equal_cost_paths=True,
        shuffle=False)
    assert len(solutions) == 2
    assert solutions[0][0] == [('A', 2), ('B', 2), ('C', 0)]
    assert solutions[1][0] == [('B', 2), ('A', 2), ('C', 0)]


def test_astar_return_all_equal_cost_paths_single_solution():
    problem = Blockworld(
        (('D', 'A'), ('C', 'B'), ()),
        [
            Blockworld.make_above_predicate(top, bottom)
            for (top, bottom) in [('C', 'D'), ('B', 'C'), ('A', 'B')]
        ],
    )
    next_goal, goal_test, h = agent.make_ordered_k_goal_cost_heuristic(
        problem, problem.initial, k=2, debug=True)
    solutions = agent.A_Star(
        problem,
        h,
        start=problem.initial,
        goal_test=goal_test,
        return_all_equal_cost_paths=True,
        shuffle=False)
    assert len(solutions) == 1
    assert solutions[0][0][0] == ('A', 2)


def test_compute_action_path_probabilities():
    assert agent.compute_action_path_probabilities([
        ('A', 'B', 'C', 'D'),
        ('A', 'B', 'C', 'D*'),
        ('A', 'B*', 'C', 'D'),
        ('A', 'B+', 'C', 'D'),
        ('Z', 'Y', 'X', 'F'),
    ]) == {
        ('A', 'B', 'C', 'D'): 1/12,
        ('A', 'B', 'C', 'D*'): 1/12,
        ('A', 'B*', 'C', 'D'): 1/6,
        ('A', 'B+', 'C', 'D'): 1/6,
        ('Z', 'Y', 'X', 'F'): 1/2,
    }
