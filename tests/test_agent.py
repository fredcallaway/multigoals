from simple_spatial import SimpleSpatial
from agent import solve_using_ordered_goal_subset, solve_using_ordered_goal_subset_perf1

# This task is solved suboptimally unless you consider all 3 goals.
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
    history, completed = solve_using_ordered_goal_subset(spatial_task_1, k=2, shuffle=False)
    assert completed
    actions = [a for a, s in history]
    assert len(actions) == 9
    assert actions == spatial_task_actions


def test_solve_using_ordered_goal_subset_perf1():
    history, completed = solve_using_ordered_goal_subset_perf1(spatial_task_1, k=2, shuffle=False)
    assert completed
    actions = [a for a, s in history]
    assert len(actions) == 9
    assert actions == spatial_task_actions
