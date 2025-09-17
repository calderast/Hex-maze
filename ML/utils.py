import sys
# import torch
import pandas as pd
sys.path.append("..")  # Use sys to add the parent directory (where src/hexmaze lives) to the path
from src.hexmaze import create_empty_hex_maze, maze_to_graph
# from src.hexmaze import maze_to_string, maze_to_barrier_set, maze_to_graph, plot_hex_maze


ADJACENCY_LIST = {}

def create_adjacency_list()->None:
    if len(ADJACENCY_LIST) == 0:
        empty_maze = create_empty_hex_maze()
        maze_as_graph = maze_to_graph(empty_maze)
        print(f"Here is the maze as a {type(maze_as_graph)}: '{maze_as_graph}'")
        for u, v in maze_as_graph.edges():
            if u not in ADJACENCY_LIST:
                ADJACENCY_LIST[u] = set()
            if v not in ADJACENCY_LIST:
                ADJACENCY_LIST[v] = set()
            ADJACENCY_LIST[u].add(v)
            ADJACENCY_LIST[v].add(u)
            print(f"{u} <-> {v}")
    print(ADJACENCY_LIST)
    
def make_dataset(save_dir: str, db_path: str = '../Maze_Databases/maze_configuration_database.pkl', train_test_split: float = 0.9)->None:
    create_adjacency_list()
    # make tokenizer
    # 
    
    
    # load in pandas dataframe
    maze_database = pd.read_pickle(db_path)
    # split into train and test
    train, test = maze_database.iloc[:int(len(maze_database)*0.9)], maze_database.iloc[int(len(maze_database)*0.9):]
    print(train.columns)
    ## print the first 50 columns in the field 'optimal_paths_all'
    # print(train.iloc[:50]['optimal_paths_all'])
    ## what is the correct way to iterate over the number of rows in a pd dataframe
    # for i in range(train.shape[0]):
    for i in range(5):
        for path in train.iloc[i]['optimal_paths_all']:
            # print(path)
            
            # inject adjacency lists into path
            prompt = []
            # is this barriers thing already a set? 
            for element in path: 
                next_moves = ADJACENCY_LIST[element] - train.iloc[i]['barriers']
                prompt.append((element, next_moves))
            print(f"#####################\nbarriers: {train.iloc[i]['barriers']}\npath: {path}\nprompt: {prompt}")
            # print(f"prompt: {prompt}, barriers: {train.iloc[i]['barriers']}")
            # tokenize path
            
            

    
    # for each maze in the desired split
        # extract the list of optimal paths
        # for each path
            # node for node in adj_list if node is not blocked
            # you'll also want to tokenize this 
            # append tokenized tensor to dataset (probably will want frozenset representation of maze for eval purposes also)
    # save the big tensor to some torch file so you can load it back in at train time


if __name__ == "__main__":
    # create_adjacency_list()
    make_dataset(save_dir = './data.pkl')
    
    