"""
Output formatting utility for search results
"""

def format_output(filename, method, goal, nodes_created, path):
    """
    Format search results according to required output format
    
    Args:
        filename (str): Problem file name
        method (str): Search method used
        goal (int): Goal node reached
        nodes_created (int): Number of nodes created during search
        path (list): Path from origin to goal
        
    Returns:
        str: Formatted output string
    """
    # Convert path to string format
    path_str = " ".join(map(str, path))
    
    # Format the output according to requirements
    output = f"{filename} {method}\n{goal} {nodes_created}\n{path_str}"
    
    return output