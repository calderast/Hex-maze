#  This folder contains many databases of good maze sequences for barrier change experiments.

### Mazes differ by the movement of a single barrier
In all cases, each maze in a sequence must be different from the previous maze by the movement of a single barrier. We also want the environment to change such that the rat will adapt their behavior in some way (i.e. a barrier movement needs to meaningfully change the structure of the maze).

### The maze structure changes when a barrier is moved
To ensure the maze structure changes, we currently have 2 additional criteria for valid mazes created by a barrier change:
1. At least one path between reward ports must be longer AND one must be shorter.
2. The optimal path order must have changed (the pair of reward ports that used to be the closest together or furthest apart is now different).

These criteria are not mutually exclusive. For example, consider a starting maze with path lengths 15, 19, 23. A new maze with path lengths 17, 19, 21 satisfies criteria 1 but not criteria 2. Conversely, a new maze with path lengths 15, 19, 17 satisfies criteria 2 but not criteria 1. A new maze with path lengths 19, 17, 23 satisfies both criteria. 

When generating a barrier sequence, we can use the `criteria_type` argument to choose if new mazes must satisfy both (`"ALL"`) or either (`"ANY"`) of these criteria.

### All mazes in a sequence are sufficiently different
Each barrier set must be different enough from the previous barrier set, AND also different enough from all other previous mazes in the sequence.

To do this, we can set a threshold `min_hex_diff`: this is the combined minimum number of hexes that need to be different on optimal paths to reward ports between ALL mazes in a sequence.


## The databases in this folder were all generated with slightly different criteria.

See [`Generate_Custom_Barrier_Sequence_Database.ipynb`](../Tutorials/Generate_Custom_Barrier_Sequence_Database.ipynb) in the `Tutorials` folder for more info.

-----


`barrier_sequence_database` contains 3126 barrier sequences. This is a good place to start.
- Sequences are 4-6 mazes long
- Every barrier change results in at least one path getting shorter and one getting longer, AND the optimal path order changes (criteria_type=ALL)
- There are at least 9 hexes different combined across all optimal paths for all mazes in a sequence (min_hex_diff=9)

`barrier_sequences_starting_from_all_mazes` contains 55896 barrier sequences, each starting from a different maze in the `maze_configuration_database`.
- Some of these "sequences" contain only one maze because no good barrier changes were found. These mazes are still included as they make good candidates for probability change experiments where barrier changes are not needed.
- barrier_sequence_database (above) is the subset of this database where sequence_length >= 4

`long_barrier_sequences` contains 438 long barrier sequences (generated by allowing get_barrier_sequence to make up to 200 recursive calls instead of the default 40).
- Sequences are 6-7 mazes long
- Every barrier change results in at least one path getting shorter and one getting longer, AND the optimal path order changes (criteria_type=ALL)
- There are at least 9 hexes different combined across all optimal paths for all mazes in a sequence (min_hex_diff=9)

`single_choice_point` contains 3720 barrier sequences where all mazes in the sequence have a single choice point.
- All sequences are at least 3 mazes long
- 104 sequences are 6 mazes long
- Every barrier change results in at least one path getting shorter and one getting longer, AND the optimal path order changes (criteria_type=ALL)
- There are at least 9 hexes different combined across all optimal paths for all mazes in a sequence (min_hex_diff=9)

`criteria_type_any_long_sequences` contains 1734 barrier sequences 5+ mazes long, with relaxed criteria (criteria_type=ANY) to generate longer sequences.
- All sequences are at least 5 mazes long
- 1338 sequences are 6+ mazes long, 564 are 7+, 334 are 8+, 220 are 9
- Every barrier change results in at least one path getting shorter and one getting longer, OR the optimal path order changes (criteria_type=ANY)
- There are at least 16 hexes different combined across all optimal paths for all mazes in a sequence (min_hex_diff=16)

`criteria_type_any_starting_from_all_mazes` contains 55896 barrier sequences, each starting from a different maze in the `maze_configuration_database`, with relaxed criteria (criteria_type=ANY) to generate longer sequences.
- criteria_type_any_long_sequences (above) is the subset of this database where sequence_length >= 5
- Some of these "sequences" contain only one maze because no good barrier changes were found. These mazes are still included as they make good candidates for probability change experiments where barrier changes are not needed.
- Every barrier change results in at least one path getting shorter and one getting longer, OR the optimal path order changes (criteria_type=ANY)
- There are at least 16 hexes different combined across all optimal paths for all mazes in a sequence (min_hex_diff=16)

`1_choice_point_all_path_lengths_different_first5000` contains 112 barrier sequences where all mazes have a single choice point AND all 3 path lengths are different.
- All sequences are at least 3 mazes long. 6 sequences are 4 mazes long
- This was generated starting from the first 5000 mazes in the `maze_configuration_database`. To add to it, start from maze 5001


As I add new databases, I will document them here (and if you generate a new database, please document it here as well!)