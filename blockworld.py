from problem import MultiProblem


class Blockworld(MultiProblem):
    '''
    Blockworld is a problem where you stack blocks, one at a time.

    Blockworld is a world with a set of blocks and a number of columns the blocks
    can be in. Blocks on top of a column can be moved to other columns. The
    goals in Blockworld often have to do with placing the blocks in order.

    The state is represented as a list of columns, where the items are listed in
    spatially ascending order. For example, this board state with 2 columns
    (A standing alone and B on top of C):

    ...
    .B.
    AC.

    is represented by this state:

    (('A',), ('C', 'B'), ())

    >>> s = (('B',), ('C', 'A'), ())
    >>> w = Blockworld(s, s)
    >>> w.result(s, ('A', 2))
    (('B',), ('C',), ('A',))
    >>> w = Blockworld(s, s, canonicalize_states=True)
    >>> w.initial
    (('C', 'A'), ('B',), ())
    >>> w.result(s, ('A', 2))
    (('A',), ('B',), ('C',))
    '''

    def __init__(self, initial, goals, canonicalize_states=False):
        if canonicalize_states:
            initial = _canonicalize_blockworld_state(initial)
        super().__init__(initial, goals)
        self.canonicalize_states = canonicalize_states
        # assert len(initial) == len(goal),\
        #    'In Blockworld, initial state and goal state must have the same number of places blocks can go.'

    def actions(self, state):
        """Returns set of actions that can be taken from this state.

        Actions are encoded as a pair. The first element of the pair
        is the label of the block that will be moved. The second element
        of the pair is the index of the column the block will be moved to.

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
        """Returns next state following action taken at current state.

        >>> s = (('B',), ('A', 'C'), ())
        >>> w = Blockworld(s, s)
        >>> w.result(s, ('B', 1))
        ((), ('A', 'C', 'B'), ())
        >>> w.result(s, (0, 1)) # HACK adding on this variant of action representation
        ((), ('A', 'C', 'B'), ())
        """
        source_block, dest_col_idx = action
        # HACK for performance reasons, we allow source_block to be an index, which avoids searching through columns.
        if isinstance(source_block, int):
            source_col_idx = source_block
            source_block = state[source_col_idx][-1]
        else:
            source_col_idx = next(colidx for colidx, col in enumerate(state) if col and col[-1] == source_block)
        result = tuple(
            col[:-1] if colidx == source_col_idx else
            col + (source_block,) if colidx == dest_col_idx else
            # We can use the same tuple from previous state since tuples are immutable.
            col
            for colidx, col in enumerate(state)
        )
        if self.canonicalize_states:
            result = _canonicalize_blockworld_state(result)
        return result

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

    @classmethod
    def make_is_bottom_of_column_predicate(cls, block):
        '''Returns a predicate that returns true when the block is the bottom block in some column.

        >>> p = Blockworld.make_is_bottom_of_column_predicate('A')
        >>> p((('A', 'C'), ('B',)))
        True
        >>> p((('B',), ('A', 'C'), ()))
        True
        >>> p((('B', 'A'), ('C',)))
        False
        '''
        def pred(state):
            return any(s[0] == block for s in state if s)

        return pred


def _canonicalize_blockworld_state(state):
    '''
    We canonicalize states so that taller columns are first, and ties between columns of same height are broken
    by ordering using the topmost symbol.
    >>> _canonicalize_blockworld_state((('A',), ('B', 'C'), ()))
    (('B', 'C'), ('A',), ())
    >>> _canonicalize_blockworld_state(((), ('D', 'A',), ('B', 'C')))
    (('D', 'A'), ('B', 'C'), ())
    >>> _canonicalize_blockworld_state((('B',), ('A',)))
    (('A',), ('B',))
    '''
    return tuple(sorted(
        state,
        key=lambda column: (-len(column), column[-1] if column else None)))


if __name__ == '__main__':
    # HACK how should we deal with open spaces? Should we permit an unlimited number?
    # HACK In some more complex cases, I think limiting the number of open spaces is an interesting constraint.
    problem = Blockworld(
        (('A', 'C'), ('B',), ()),
        (
            Blockworld.make_above_predicate('B', 'C'),
            Blockworld.make_above_predicate('A', 'B'),
        ),
    )

    # We are enumerating all states
    def _visit(problem):
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
        return seen
    seen = _visit(problem)
    print('Total number of states:', len(seen))
    assert len(seen) == 60

    # Implementing state canonicalization
    problem_canonicalize = Blockworld(((), ('A', 'C'), ('B',)), problem.goals, canonicalize_states=True)
    canonicalized = _visit(problem_canonicalize)
    print('Total unique number of states:', len(canonicalized))
    assert len(canonicalized) == 13

    import doctest
    fail_count, test_count = doctest.testmod()
    if not fail_count:
        print('\n\t** All {} tests passed! **\n'.format(test_count))
