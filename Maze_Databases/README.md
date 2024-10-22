#  This folder contains various databases of different mazes.

These mazes are useful for training or probability change experiments.

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