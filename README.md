## This repo provides a database of possible hex maze configurations and sequences of barrier change configurations.

### Database of hex maze configurations
`maze_configuration_database` contains 55,896 possible hex maze configurations with the following attributes:
- 9 barriers
- no unreachable hexes
- path lengths between reward ports are between 15-25 hexes
- all critical choice points are >=6 hexes away from a reward port
- there are a maximum of 3 critical choice points
- there are no straight paths >6 hexes to a reward port (including port hex)
- there are no straight paths >6 hexes in the middle of the maze

This database also provides information about each maze configuration:
- length of the optimal path(s) between each pair of reward ports
- lists of hexes defining the optimal path(s) between those reward ports
- number of choice points, and which hexes are choice points
- number of cycles (where the rat can start and end at the same hex without turning around) and lists of hexes defining those cycles
- a set of other mazes isomorphic to this maze (representing reflections and rotations of the maze)

This database is provided in both csv (.csv) and pickle (.pkl) format - csv is better to explore in excel, but pickle is preferable for loading and working with in jupyter notebooks.

### Database of barrier sequences
`Barrier_Sequence_Databases/` contains multiple databases of barrier sequences (consecutive maze configurations that differ by the movement of a single barrier).
I'm currently still updating/adding to these databases, but `barrier_sequences_first1000` contains 1000 possible barrier sequences and is a good place to start for now.
