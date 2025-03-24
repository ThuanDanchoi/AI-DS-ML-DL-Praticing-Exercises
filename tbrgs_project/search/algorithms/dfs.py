"""
Depth-First Search (DFS)  
"""

def search(graph):
    """
    Perform Depth-First Search on the graph
    
    Args:
        graph (Graph): The graph to search
        
    Returns:
        tuple: (goal_node, nodes_created, path)
            - goal_node: ID of destination reached
            - nodes_created: Number of nodes created during search
            - path: List of nodes in path from origin to goal
    """
    origin = graph.origin
    stack = [(origin, [origin])]
    visited = set()
    nodes_created = 1

    while stack:
        node, path = stack.pop()

        if node in visited:
            continue
        
        visited.add(node)

        if graph.is_destination(node):
            return (node, nodes_created, path)
        
        neighbors = graph.get_neighbors(node)
        neighbors.reverse()

        for neighbor, _ in neighbors:
            if neighbor not in visited:
                new_path = path + [neighbor]
                stack.append((neighbor, new_path))
                nodes_created += 1

    return None, nodes_created, []

