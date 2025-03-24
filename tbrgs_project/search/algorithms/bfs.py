"""
Breadth-First Search (BFS)  
"""
from collections import deque

def search(graph):
    """
    Perform Breadth-First Search on the graph
    
    Args:
        graph (Graph): The graph to search
        
    Returns:
        tuple: (goal_node, nodes_created, path)
            - goal_node: ID of destination reached
            - nodes_created: Number of nodes created during search
            - path: List of nodes in path from origin to goal
    """
    origin = graph.origin
    queue = deque([(origin, [origin])])

    visited = set()
    nodes_created = 1

    while queue:
        node, path = queue.popleft()

        if graph.is_destination(node):
            return node, nodes_created, path
        
        neighbors = graph.get_neighbors(node)

        for neighbor, _ in neighbors:
            if neighbor not in visited:
                new_path = path + [neighbor]
                queue.append((neighbor, new_path))
                visited.add(neighbor)
                nodes_created += 1

    return None, nodes_created, []