# Hex-maze
This repo provides a set of functions to generate, plot, and calculate info about hex maze configurations and optimal barrier change sequences for the hex maze behavioral task used by the Berke Lab at UCSF. It also provides databases of valid maze configurations and their attributes.

## Step 1. Fork and clone the repository
To contribute to this project, you first need to fork the repository to your own GitHub account. This creates a copy of the project where you can make changes.
Do this by clicking the "Fork" button at the top-right corner of the repository page and following the instructions.

Once you have forked the repository, you need to clone it to your local machine to start working on it.
1. Open a terminal or command prompt.
2. Clone your forked repository using the following command (replacing {your-github-username} with your username):

    ```sh
    git clone https://github.com/{your-github-username}/Hex-maze.git

4. Navigate to the newly created `Hex-maze` directory.

    ```sh
    cd Hex-maze

## Step 2. Install dependencies
1. First, make sure you are in the repo inside a terminal or command prompt (Step 3 above). 

2. To install all necessary dependencies for this project, run the following command:

    ```sh
    pip install -r requirements.txt

## Step 3. Start the tutorials
Navigate to the `Tutorials/` folder and begin with the [`Getting_Started.ipynb`](Tutorials/Getting_Started.ipynb) notebook.

`Tutorials/` also includes the following tutorial notebooks:
- [Plotting hex mazes and barrier change sequences](Tutorials/Plotting_Hex_Mazes.ipynb)
- [Searching the maze configuration database for the mazes you want](Tutorials/Maze_Configuration_Database_Search.ipynb)
- [Searching the barrier sequence database for the sequence you want](Tutorials/Barrier_Sequence_Database_Search.ipynb)
- [Demos of useful hex maze functions](Tutorials/Hex_Maze_Functions.ipynb)
- [Generating barrier sequences with custom criteria](Tutorials/Generate_Custom_Barrier_Sequence_Database.ipynb)

These 3 are also provided for reference:
- [How the hex maze database was generated](Tutorials/Generate_Hex_Maze_Database.ipynb)
- [How the probability change database was generated](Tutorials/Generate_Probability_Change_Database.ipynb)
- [How the training maze database was generated](Tutorials/Generate_Training_Maze_Database.ipynb)

Note that some of these tutorials are currently in progress, LMK if you need one asap and I'll make it a priority!

## Step 4: Explore and use the databases!
This repo provides the following databases of valid maze configurations and barrier change sequences for the hex maze task:

### Database of hex maze configurations
`Maze_Databases/maze_configuration_database` contains 55,896 possible hex maze configurations with the following attributes:
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

This database is provided in both csv (.csv) and pickle (.pkl) format - csv is better to explore in excel, but pickle is preferable for loading and working with in jupyter notebooks (csv tends to load everything as a string).

This database was generated using the `Generate_Hex_Maze_Database.ipynb` notebook available in the `Tutorials/` folder.

### Database of mazes for barrier change experiments
The `Barrier_Sequence_Databases/` folder contains multiple databases of barrier sequences (consecutive maze configurations that differ by the movement of a single barrier).

- `barrier_sequence_database` contains 3126 barrier sequences that are 4-6 mazes long, and is a good place to start. 
- `single_choice_point` contains 3720 barrier sequences >= 3 mazes long where all mazes in the sequence have a single choice point.

The `Barrier_Sequence_Database_Search.ipynb` notebook in the `Tutorials/` folder has more information about the other available databases, and how to search these databases for a barrier sequence that fits your criteria.

Custom databases can be generated using the `Generate_Custom_Barrier_Sequence_Database.ipynb` notebook available in the `Tutorials/` folder.

### Database of mazes for probability change experiments
`Maze_Databases/probability_change_mazes` contains a database of mazes good for probability change experiments. These mazes are grouped such that all mazes in a group differ by at least 10 hexes on optimal paths.

This database was generated using the `Generate_Probability_Change_Database.ipynb` notebook available in the `Tutorials/` folder.

### Database of mazes for early stages of training
`Maze_Databases/training_maze_database` contains a database of mazes good for training. There are 5-6 barriers and all paths are the same length (either 15 or 17 hexes).

This database was generated using the `Generate_Training_Maze_Database.ipynb` notebook available in the `Tutorials/` folder.

## Other info

### Hex maze functions
`hex_maze_utils.py` provides all of the functions for hex maze related tasks. 

If you'd like any extra functionality, let me know (or feel free to add it and submit a PR)!

A tutorial for the most useful functions can be found at `Tutorials/Hex_Maze_Functions.ipynb`. 
For functions without tutorials, you can view the documentation running `help(function_name)`, or just by scrolling through the `hex_maze_utils.py` file.
