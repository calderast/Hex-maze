# Tutorial Notebooks

The tutorial notebooks have two-digits in their names, the first of which indicates its 'batch', as
described in the categories below.

## 0. Intro

Everyone should start with the [Getting Started](./00_Getting_Started.ipynb) notebook. 
This gives an overview of how to work with the hex maze, as well as descriptions of the other tutorial notebooks.

## 1. Analysis and plotting

These notebooks cover analysis ([Hex Maze Functions](./10_Hex_Maze_Functions.ipynb)) and plotting ([Plotting Hex Mazes](./11_Plotting_Hex_Mazes.ipynb)). This notebook ([Compare Maze Paths](./12_Compare_Maze_Paths.ipynb)) covers path choice analysis specific to barrier change sessions.
Reference these notebooks if you are working with hex maze data.

## 2. Searching maze databases

These notebooks cover loading and filtering the maze databases to find optimal configurations for your experiments. 
Reference these notebooks if you are running experiments using the hex maze.
See [Maze Configuration Database Search](./20_Maze_Configuration_Database_Search.ipynb) and [Barrier Sequence Database Search](./21_Barrier_Sequence_Database_Search.ipynb).

## 3. Generating maze databases 

These notebooks are provided as reference for how the maze databases were generated.
You can ignore these unless you plan to generate a new database using different maze criteria.

## 4. Modeling port values

This notebook ([Port Value Models](./40_Port_Value_Models.ipynb)) covers various reinforcement learning models used to fit port values (reward expectation) based on rat port choices and reward outcomes. Note the RL module is still in progress.