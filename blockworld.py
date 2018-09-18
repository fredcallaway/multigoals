from problem import MultiProblem


class Blockworld(MultiProblem):
    def __init__(self, initial, goals):
        super().__init__(initial, goals)
        # assert len(initial) == len(goal),\
        #    'In Blockworld, initial state and goal state must have the same number of places blocks can go.'

    def actions(self, state):
        """
        >>> s = (('A', 'C'), ('B',), ())
        >>> w = Blockworld(s, s)
        >>> w.actions(s)
        [('C', 1), ('C', 2), ('B', 0), ('B', 2)]
        """
        return [
            (source_col[-1], dest_col_idx)
            for source_col_idx, source_col in enumerate(state)
            # We can only take from a column with some blocks.
            if source_col
            for dest_col_idx in range(len(state))
            # We don't want to permit moving to the same column.
            if source_col_idx != dest_col_idx
        ]

    def result(self, state, action):
        """
        >>> s = (('A', 'C'), ('B',), ())
        >>> w = Blockworld(s, s)
        >>> w.result(s, ('B', 0))
        (('A', 'C', 'B'), (), ())
        """
        # TODO should we canonicalize this state in some way?
        # Would make it easier to avoid repeating a state in search
        source_block, dest_col_idx = action
        source_col_idx = next(colidx for colidx, col in enumerate(state) if col and col[-1] == source_block)
        return tuple(
            col[:-1] if colidx == source_col_idx else
            col + (source_block,) if colidx == dest_col_idx else
            # We can use the same tuple from previous state since tuples are immutable.
            col
            for colidx, col in enumerate(state)
        )

    def render(self, state):
        """Returns a graphical depiction of the state as a string.

        >>> s = (('A', 'C'), ('B',))
        >>> w = Blockworld(s, s)
        >>> w.render(s)
        '..\\nC.\\nAB\\n'
        """
        max_height = sum(len(col) for col in state)
        result = ''
        for colidx in reversed(range(max_height)):
            result += ''.join(col[colidx] if colidx < len(col) else '.' for col in state) + '\n'
        return result

    @classmethod
    def _findblock(cls, state, block):
        '''Returns the column index and row index for a block in a state.
        '''
        return next(
            (colidx, rowidx)
            for colidx, col in enumerate(state)
            for rowidx, cell in enumerate(col)
            if cell == block)

    @classmethod
    def make_above_predicate(cls, top, bottom):
        '''Returns a predicate that returns true when the top block is above the bottom block in the same column.

        >>> p = Blockworld.make_above_predicate('A', 'B')
        >>> p((('A', 'C'), ('B',)))
        False
        >>> p((('A', 'B'), ('C',)))
        False
        >>> p((('B', 'A'), ('C',)))
        True
        '''
        def pred(state):
            top_col, top_row = cls._findblock(state, top)
            bottom_col, bottom_row = cls._findblock(state, bottom)
            return top_col == bottom_col and top_row - 1 == bottom_row

        return pred


if __name__ == '__main__':
    # HACK how should we deal with open spaces? Should we permit an unlimited number?
    # HACK In some more complex cases, I think limiting the number of open spaces is an interesting constraint.
    problem = Blockworld(
        (('A', 'C'), ('B',), ()),
        (
            Blockworld.make_above_predicate('A', 'B'),
            Blockworld.make_above_predicate('B', 'C'),
        ),
    )

    # We are enumerating all states
    seen = set()
    q = [problem.initial]
    while q:
        s = q.pop()
        if s in seen:
            continue
        seen.add(s)
        for a in problem.actions(s):
            next_state = problem.result(s, a)
            if next_state not in seen:
                q.append(next_state)
    print('Total number of states:', len(seen))

    # Implementing state canonicalization
    def _canonicalize_blockworld_state(state):
        # We canonicalize states so that taller columns are first, and ties between columns of same height are broken
        # by ordering using the topmost symbol.
        return tuple(sorted(
            state,
            reverse=True,
            key=lambda column: (len(column), column[-1] if column else None)))
    canonicalized = set(_canonicalize_blockworld_state(s) for s in seen)
    print('Total unique number of states:', len(canonicalized))

    import doctest
    fail_count, test_count = doctest.testmod()
    if not fail_count:
        print('\n\t** All {} tests passed! **\n'.format(test_count))
