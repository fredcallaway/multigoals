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
