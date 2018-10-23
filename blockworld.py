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
    >>> Blockworld(s, s).result(s, ('A', 2))
    (('B',), ('C',), ('A',))
    >>> w = Blockworld(s, s, canonicalize_states=True)
    >>> w.initial # Initial has been modified since we're canonicalizing
    (('C', 'A'), ('B',), ())
    >>> w.result(s, ('A', 2))
    (('A',), ('B',), ('C',))
    '''

    def __init__(
        self, initial, goals,
        canonicalize_states=False,
        height_limits=None,
        towers_of_hanoi=False,
    ):
        if canonicalize_states:
            initial = _canonicalize_blockworld_state(initial)
        super().__init__(initial, goals)
        self.canonicalize_states = canonicalize_states
        if height_limits is None:
            num_blocks = sum(len(col) for col in initial)
            height_limits = (num_blocks,) * len(initial)
        self.height_limits = height_limits
        assert len(self.height_limits) == len(initial), 'Must supply same number of height limits as there are spaces.'
        self.towers_of_hanoi = towers_of_hanoi
        # assert len(initial) == len(goal),\
        #    'In Blockworld, initial state and goal state must have the same number of places blocks can go.'

    def actions(self, state):
        """Returns set of actions that can be taken from this state.

        Actions are encoded as a pair. The first element of the pair
        is the label of the block that will be moved. The second element
        of the pair is the index of the column the block will be moved to.

        >>> s = (('A', 'C'), ('B',), ())
        >>> Blockworld(s, s).actions(s)
        [('C', 1), ('C', 2), ('B', 0), ('B', 2)]
        >>> Blockworld(s, s, height_limits=(2, 2, 2)).actions(s)
        [('C', 1), ('C', 2), ('B', 2)]
        >>> s = (('A',), ('B',), ('C',))
        >>> Blockworld(s, s).actions(s)
        [('A', 1), ('A', 2), ('B', 0), ('B', 2), ('C', 0), ('C', 1)]
        >>> Blockworld(s, s, towers_of_hanoi=True).actions(s)
        [('A', 1), ('A', 2), ('B', 2)]
        """
        def is_legal_move(source_col_idx, source_col, dest_col_idx, dest_col):
            if self.towers_of_hanoi:
                if dest_col:
                    # HACK we assume that python's comparators suffice to compare column elements.
                    # This will always work for numbers and single-letter strings.
                    return source_col[-1] < dest_col[-1]
                else:
                    return True
            elif self.height_limits is not None:
                # This is a legal move if we are not at a height limit
                return len(dest_col) < self.height_limits[dest_col_idx]
            else:
                # HACK this no longer happens...
                # Otherwise, any move is fine!
                return True
        return [
            (source_col[-1], dest_col_idx)
            for source_col_idx, source_col in enumerate(state)
            # We can only take from a column with some blocks.
            if source_col
            for dest_col_idx, dest_col in enumerate(state)
            # We don't want to permit moving to the same column.
            if source_col_idx != dest_col_idx
            if is_legal_move(source_col_idx, source_col, dest_col_idx, dest_col)
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
        >>> Blockworld(s, s).render(s)
        '..\\nC.\\nAB\\n'
        >>> Blockworld(s, s, height_limits=(3, 1)).render(s)
        '. \\nC \\nAB\\n'
        """
        max_height = max(self.height_limits)
        result = ''
        for rowidx in reversed(range(max_height)):
            result += ''.join(
                col[rowidx] if rowidx < len(col) else
                '.' if rowidx < self.height_limits[colidx] else ' '
                for colidx, col in enumerate(state)) + '\n'
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
        pred.__name__ = f'{top} is on top of {bottom}'

        return pred

    @classmethod
    def make_is_bottom_of_column_predicate(cls, block, column_index=None):
        '''Returns a predicate that returns true when the block is the bottom block in some column.

        Can optionally supply a specific column index to look at. This optional parameter is primarily
        used for Tower of London tasks.

        >>> p = Blockworld.make_is_bottom_of_column_predicate('A')
        >>> p((('A', 'C'), ('B',)))
        True
        >>> p((('B',), ('A', 'C'), ()))
        True
        >>> p((('B', 'A'), ('C',)))
        False
        >>> p = Blockworld.make_is_bottom_of_column_predicate('A', column_index=0)
        >>> p((('A', 'C'), ('B',)))
        True
        >>> p((('B',), ('A', 'C'), ()))
        False
        >>> p((('B', 'A'), ('C',)))
        False
        >>> p(((), ('C',)))
        False
        '''
        if column_index is None:
            def pred(state):
                return any(s[0] == block for s in state if s)
            pred.__name__ = f'{block} is at the bottom of a column'
        else:
            def pred(state):
                col = state[column_index]
                return bool(col) and col[0] == block
            pred.__name__ = f'{block} is at the bottom of column {column_index}'

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

    # Counting with state canonicalization
    problem_canonicalize = Blockworld(((), ('A', 'C'), ('B',)), problem.goals, canonicalize_states=True)
    canonicalized = _visit(problem_canonicalize)
    print('Total unique number of states:', len(canonicalized))
    assert len(canonicalized) == 13

    # Counting with ToH
    problem_canonicalize = Blockworld(((), ('C', 'A'), ('B',)), problem.goals, towers_of_hanoi=True, canonicalize_states=True)
    canonicalized = _visit(problem_canonicalize)
    print('Total unique number of states for ToH:', len(canonicalized))

    # Counting with ToL height limits
    problem_canonicalize = Blockworld(((), ('A', 'C'), ('B',)), problem.goals, height_limits=(3, 2, 1), canonicalize_states=True)
    canonicalized = _visit(problem_canonicalize)
    print('Total unique number of states for ToL with height limits:', len(canonicalized))

    import doctest
    fail_count, test_count = doctest.testmod()
    if not fail_count:
        print('\n\t** All {} tests passed! **\n'.format(test_count))
