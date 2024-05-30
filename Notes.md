

We represent each maze as a frozenset of barriers 
(frozensets are immutable sets - we need this because we can't store mutable objects in a set)

This is also why we often make lists into tuples (tuples are immutable/hashable)