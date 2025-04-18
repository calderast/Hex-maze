#  This folder contains various databases of different mazes.

`maze_configuration_database` is the main maze database. It contains 55,896 possible hex maze configurations with the following attributes:
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

-----

We also provide databases of mazes specifically tailored to training or probability change experiments. For barrier change experiments, see the [`Barrier_Sequence_Databases`](../Barrier_Sequence_Databases/README.md) folder.

-----

`probability_change_mazes` contains 38208 mazes that are currently not used in barrier sequences. These are good for probabilty change experiments.
- The column `probability_group` groups the mazes, such that all mazes in a group have at least 10 hexes different on optimal paths.
- There are 2077 groups with ~5-40 mazes in each group.
- See [`Generate_Probability_Change_Database.ipynb`](../Tutorials/Generate_Probability_Change_Database.ipynb) in the `Tutorials` folder for more info on how this database was generated.

`training_maze_database` contains 11554 mazes used for early stages of training when the rats are getting used to the maze.
- All path lengths between reward ports are of equal length. There are 9288 mazes where all reward path lengths are 15 hexes, and 2266 mazes where all reward path lengths are 17 hexes.
- Mazes have either 5 or 6 barriers.
- There are no straight paths >8 hexes long
- See [`Generate_Training_Maze_Database.ipynb`](../Tutorials/Generate_Training_Maze_Database.ipynb) in the `Tutorials` folder for more info on how this database was generated.