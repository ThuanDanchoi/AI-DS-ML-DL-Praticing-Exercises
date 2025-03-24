"""
Search program for Traffic-based Route Guidance System - Part A
Usage: python search.py <filename> <method>
"""

import sys
import os
from utils.file_parser import parse_problem_file
from utils.output import format_output
from graph import Graph

# Import search algorithms
from algorithms.dfs import search as dfs_search
from algorithms.bfs import search as bfs_search
from algorithms.gbfs import search as gbfs_search
from algorithms.astar import search as astar_search

def main():
    # Check command line arguments
    if len(sys.argv) != 3:
        print("Usage: python search.py <filename> <method>")
        sys.exit(1)
    
    filename = sys.argv[1]
    method = sys.argv[2].upper()
    
    # Check if file exists
    if not os.path.isfile(filename):
        print(f"Error: File '{filename}' not found")
        sys.exit(1)
    
    # Check if method is valid
    valid_methods = ['DFS', 'BFS', 'GBFS', 'AS']
    if method not in valid_methods:
        print(f"Error: Method '{method}' not recognized. Choose from: {', '.join(valid_methods)}")
        sys.exit(1)
    
    # Parse problem file
    nodes, edges, origin, destinations = parse_problem_file(filename)
    
    # Create graph
    graph = Graph(nodes, edges, origin, destinations)
    
    # Run the specified search algorithm
    if method == 'DFS':
        goal, nodes_created, path = dfs_search(graph)
    elif method == 'BFS':
        goal, nodes_created, path = bfs_search(graph)
    elif method == 'GBFS':
        goal, nodes_created, path = gbfs_search(graph)
    elif method == 'AS':
        goal, nodes_created, path = astar_search(graph)
    
    # Print the results in the required format
    if goal is not None:
        output = format_output(filename, method, goal, nodes_created, path)
        print(output)
    else:
        print(f"{filename} {method}\nNo solution found.")

if __name__ == "__main__":
    main()