from simple_spatial import SimpleSpatial
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
