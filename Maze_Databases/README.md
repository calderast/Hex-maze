#  This folder contains various databases of different mazes.

Currently, it only has one database of mazes good for probability change experiments! 

-----


`probability_change_mazes` contains 38208 mazes that are currently not used in barrier sequences. These are good for probabilty change experiments.
- The column `probability_group` groups the mazes, such that all mazes in a group have at least 10 hexes different on optimal paths.
- There are 2077 groups with ~5-40 mazes in each group.
- See [`Generate_Probability_Change_Database.ipynb`](../Tutorials/Generate_Probability_Change_Database.ipynb) in the `Tutorials` folder for more info on how this database was generated.