from problem import MultiProblem


class SimpleSpatial(MultiProblem):
    action_to_label = {
        (-1, 0): 'left',
        (1, 0): 'right',
        (0, -1): 'up',
        (0, 1): 'down',
    }

    def __init__(self, initial, map, goals):
        '''
        Map format is list of strings, where the strings are rows in the map. We assume the map is rectangular.
        The map has some milestones which correspond to our goals. They are marked locations that we aim to visit.
        The map's origin is in the upper left.

        State format is a tuple with two elements.
        First element is the (x index, y index) pair that will be used to index into the map.
        Second element is a tuple that contains all milestones that have been visited.
        '''
        super().__init__(initial, goals)
        self.map = map
        width = len(self.map[0])
        assert all(len(row) == width for row in self.map), 'Found row in map with invalid width: {}'.format(self.map)

    def actions(self, state):
        """
        >>> map = ["...", "...", "..."]
        >>> w = SimpleSpatial(((1, 1), ()), map, None)
        >>> w.actions(w.initial)
        [(-1, 0), (1, 0), (0, -1), (0, 1)]
        >>> w.actions(((0, 0), ()))
        [(1, 0), (0, 1)]
        >>> w.actions(((2, 2), ()))
        [(-1, 0), (0, -1)]
        """
        width, height = len(self.map[0]), len(self.map)
        (x, y), milestones = state
        valid_actions = []
        if 0 < x:
            valid_actions.append((-1, 0))
        if x < width - 1:
            valid_actions.append((1, 0))
        if 0 < y:
            valid_actions.append((0, -1))
        if y < height - 1:
            valid_actions.append((0, 1))
        return valid_actions

    def result(self, state, action):
        """
        >>> map = ["...", "A..", "..."]
        >>> w = SimpleSpatial(((1, 1), ()), map, None)
        >>> w.result(w.initial, (-1, 0))
        ((0, 1), ('A',))
        """
        (x, y), milestones = state
        # Update based on the action
        x, y = x + action[0], y + action[1]
        # If there is a milestone, then we add that to our list of visited milestones.
        if self.map[y][x] not in ('.', ' '):
            milestones = milestones + (self.map[y][x],)
        return (x, y), milestones

    def render(self, state):
        """Returns a graphical depiction of the state as a string.

        >>> map = ["...", "A..", "..."]
        >>> w = SimpleSpatial(((1, 1), ()), map, None)
        >>> w.render(w.initial)
        '...\\nA@.\\n...\\n'
        >>> w.render(((0, 0), ()))
        '@..\\nA..\\n...\\n'
        >>> w.render(((0, 1), ()))
        '...\\n@..\\n...\\n'
        """
        (x, y), milestones = state
        result = ''
        for rowidx, row in enumerate(self.map):
            # HACK find a way to show both current position and milestone we're on?
            result += ''.join(
                '@' if colidx == x and rowidx == y else cell
                for colidx, cell in enumerate(row)
            ) + '\n'
        return result

    @classmethod
    def make_visited_goal(cls, goal_milestone):
        '''
        Creates a predicate that checks to see if a milestone was visited.
        '''
        def pred(state):
            _, visited_milestones = state
            return goal_milestone in visited_milestones
        pred.__name__ = f'Visited milestone {goal_milestone}'
        return pred


if __name__ == '__main__':
    import doctest
    fail_count, test_count = doctest.testmod()
    if not fail_count:
        print('\n\t** All {} tests passed! **\n'.format(test_count))
