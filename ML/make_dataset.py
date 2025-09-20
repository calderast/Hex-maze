import sys
import argparse
import torch
import time
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
    
def make_dataset(save_dir: str, split: str, db_path: str = '../Maze_Databases/maze_configuration_database.pkl', train_test_split: float = 0.9, n_mazes_to_process: int = None)->None:
    max_len = 128
    pad_id = 0
    dtype = torch.int32
    print(f"going to save to {save_dir}")
    
    # make tokenizer
    # 
    
    
    # load in pandas dataframe
    maze_database = pd.read_pickle(db_path)
    # split into train and test
    if split == 'train':
        df = maze_database.iloc[:int(len(maze_database)*train_test_split)]
    else:
        df = maze_database.iloc[int(len(maze_database)*train_test_split):]

    ## print the first 50 columns in the field 'optimal_paths_all'
    # print(train.iloc[:50]['optimal_paths_all'])
    ## what is the correct way to iterate over the number of rows in a pd dataframe
    # for i in range(train.shape[0]):
    
    if n_mazes_to_process:
        N = n_mazes_to_process
    else:
        N = df.shape[0]
    tokens = torch.full((N, max_len), pad_id, dtype=torch.int32)
    
    start_time = time.time()
    for i in range(N):
        if i % 500 == 0:
            print(f"iter {i}")
        for path in df.iloc[i]['optimal_paths_all']:
            # print(path)
            # print(type(path))
            sequence = []
            for element in path:
                sequence.append(element)
                sequence.append(50)
                # unpack the set of neighbors minus barriers
                for nbr in ADJACENCY_LIST[element] - df.iloc[i]['barriers']:
                    sequence.append(nbr)
                sequence.append(50)
            # print(sequence)
            if len(sequence)>max_len:
                print(f"uh oh, max_len {max_len} exceeded by {len(sequence)}")
                sequence = sequence[:max_len]
            tokens[i,:len(sequence)] = torch.tensor(sequence, dtype=dtype)
    end_time=time.time()
    print(f"processed {N} mazes in {end_time - start_time}:.2f seconds. Now saving") 
    torch.save(tokens, save_dir)
    return 0
    

def main()->None:
    parser = argparse.ArgumentParser(
        description="Generate Train Dataset from Mazes for autoregressive transformer",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--split", type=str, required=False,
                        choices=['test','train'],
                       help='generate dataset for train or test split')
    parser.add_argument("--save_dir", type=str, required=False, default='./data.pt',
                        help="absolute or relative path to save file at (e.g './data.pt'")
    parser.add_argument('--split_prop',type=float, required=False, default=0.9,
                        help='% of data going to train split')
    parser.add_argument('--n_mazes_to_process', type=int, required=False, default=None, help='only process n mazes rather than whole split. for debug purposes only')
    # parser.add_argument("--executable", type=str, required=False, default = 'python',
    #                     help='executable command to invoke. e.g "uv run" or "python3" or "python" or "torchrun". default = python')


    # Create adjacency list
    create_adjacency_list()
    
    # parse input args
    args = parser.parse_args()
    
    # generate train/test dataset
    make_dataset(save_dir = args.save_dir, split = args.split, n_mazes_to_process = args.n_mazes_to_process)
    

if __name__ == "__main__":
    sys.exit(main())
    
    