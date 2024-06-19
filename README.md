# Hex-maze
This repo provides a set of functions to generate and plot hex maze configurations and optimal barrier change sequences for the hex maze behavioral task used by the Berke Lab at UCSF. It also provides databases of valid maze configurations and their attributes.

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
Navigate to the `Tutorials/` folder and begin with the `Getting_Started.ipynb` notebook.

`Tutorials/` also includes the following tutorial notebooks:
- [Plotting hex mazes and barrier change sequences](Tutorials/Plotting_Hex_Mazes.ipynb)
- [Searching the maze configuration database for the mazes you want](Tutorials/Maze_Configuration_Database_Search.ipynb)
- [Searching the barrier sequence database for the sequence you want](Tutorials/Barrier_Sequence_Database_Search.ipynb)
- [Demos of useful hex maze functions](Tutorials/Hex_Maze_Functions.ipynb)
- [How the hex maze database was generated](Tutorials/Generate_Hex_Maze_Database.ipynb)
- [How the barrier sequence database was generated](Tutorials/Generate_Barrier_Sequence_Database.ipynb)

Coming soon:
- Additional demos for other hex maze functions
- Generating barrier sequences with more specific criteria

Note that some of these tutorials are currently in progress, LMK if you need one asap and I'll make it a priority!

## Step 4: Explore and use the databases!
This repo provides the following databases of valid maze configurations and barrier change sequences for the hex maze task:

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

This database is provided in both csv (.csv) and pickle (.pkl) format - csv is better to explore in excel, but pickle is preferable for loading and working with in jupyter notebooks (csv tends to load everything as a string).

This database was generated using the `Generate_Hex_Maze_Database.ipynb` notebook available in the `Tutorials/` folder.

### Database of barrier sequences
The `Barrier_Sequence_Databases/` folder contains multiple databases of barrier sequences (consecutive maze configurations that differ by the movement of a single barrier).
I'm currently still updating/adding to these databases, but `barrier_sequences_first1000` contains 1000 possible barrier sequences and is a good place to start for now.

These databases are generated using the `Generate_Barrier_Sequence_Database.ipynb` notebook available in the `Tutorials/` folder.

## Other info

### Hex maze functions
`hex_maze_utils.py` provides all of the functions for hex maze related tasks. 

All of these functions are (hopefully) well documented (if not, let me know!!). 

A tutorial for the most useful functions can be found at `Tutorials/Hex_Maze_Functions.ipynb`. 
For functions without tutorials, you can view the documentation running `help(function_name)`, or just by scrolling through the `hex_maze_utils.py` file.

### dev
The `dev` folder is my sandbox/trashcan for things I'm currently working on or have abandoned. You can ignore it :)
