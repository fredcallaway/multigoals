from functools import lru_cache
import blockworld
import agent
import numpy as np


def fit_epsilon_greedy_model(model_p, rand_p, debug=False):
    '''
    Fits an epsilon greedy model given the probability of action per a model and per random action.
    Returns a dict with probability of random action under key `p_err` and log-likelihood of fit under key `logp`.

    >>> fit_epsilon_greedy_model(np.array([1, 0, 0, 0]), np.array([0, 1, 1, 1]))
    {'p_err': 0.7551020408163265, 'logp': -2.2496208047218325}
    '''
    def noisy_model_p(p_err):
        return (1 - p_err) * model_p + p_err * rand_p

    assert model_p.shape == rand_p.shape, \
        f'Shapes of probability vectors do not match. Found: {model_p.shape} {rand_p.shape}'

    # indexing trick [:, None] adds new dim to use vectorized ops via broadcasting
    p_err = np.linspace(0, 1)[:, None]
    logp = np.log(noisy_model_p(p_err))
    total_logp = logp.sum(-1)  # sum logp across trials
    assert (p_err.shape[0],) == total_logp.shape, \
        f'Shapes of p(err) and logp vectors do not match. Found: {p_err.shape} {total_logp.shape}'
    if debug:
        print('p_err.shape', p_err.shape, 'logp.shape', logp.shape, 'total_logp.shape', total_logp.shape)
    i = np.argmax(total_logp)
    best_p_err = p_err.squeeze()[i]

    if debug:
        import matplotlib.pyplot as plt
        plt.plot(p_err, total_logp)
        plt.xlabel('Error Probability')
        plt.ylabel('Log Likelihood')
        plt.axvline(best_p_err, c='r', ls='--', label='MLE Error Probability')
        plt.legend()

    return dict(p_err=best_p_err, logp=total_logp[i])


def make_goals(state):
    return blockworld.Blockworld.generate_tower_of_london_goals(state, column_index=1, debug=False)


@lru_cache(maxsize=2**14)
def optimal_num_actions(s):
    problem = blockworld.Blockworld(s, make_goals(s))
    return len(agent.bfs_search(problem))


@lru_cache(maxsize=int(2**30))
def get_actions(state, k=None, depth_limit=None):
    goals_with_clearing = make_goals(state)

    if k is None:
        k = len(goals_with_clearing)
    problem = blockworld.Blockworld(state, goals_with_clearing)
    # HACK could use next_goal to find out how far into states to go
    next_goal, goal_test, h = agent.make_ordered_k_goal_cost_heuristic(problem, state, k=k, debug=False)
    solutions = agent.A_Star(
        problem,
        h,
        start=state,
        goal_test=goal_test,
        depth_limit=depth_limit,
        return_all_equal_cost_paths=True,
        shuffle=False)
    # HACK do we just get next actions or the action sequence?
    # Returning sequence might afford some optimizations outside of this function.
    next_actions = [actions[0] for actions, states in solutions]
    # HACK we should decide how to deal with repeated actions. More repetitions means
    # more solutions are in that part of the subtree.
    return next_actions


def probability_of_action(state, participant_action, get_actions_fn_or_kwargs, uniform_action_selection=True):
    '''
    This function computes the probability that an agent might take the action a participant took.
    Can weight probability of action in different ways. Uniform action selection corresponds to unweighted
    action selection. Otherwise actions are weighted by the number of times they appear in the set of valid
    action paths.

    >>> probability_of_action(None, 1, lambda s: [0, 1, 1, 1])
    0.5
    >>> probability_of_action(None, 1, lambda s: [0, 1, 1, 1], uniform_action_selection=False)
    0.75
    >>> probability_of_action((('D', 'A'), ('C', 'B'), ()), ('A', 2), dict(k=1))
    0.5
    '''
    agent_actions = get_actions_fn_or_kwargs(state) if callable(get_actions_fn_or_kwargs) \
        else get_actions(state, **get_actions_fn_or_kwargs)
    if uniform_action_selection:
        unique_actions = set(agent_actions)
        if participant_action in unique_actions:
            return 1/len(unique_actions)
        else:
            return 0
        '''
        return {
            a: 1/len(unique_actions)
            for a in unique_actions
        }
        '''
    else:
        return sum(1 for a in agent_actions if a == participant_action)/len(agent_actions)
        '''
        return {
            a: ct/len(agent_actions)
            for a, ct in
            Counter(agent_actions).items()
        }
        '''
