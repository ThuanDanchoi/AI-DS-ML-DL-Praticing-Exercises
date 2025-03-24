"""
A* Search (A*)
"""
import heapq

def search(graph):
    """
    Perform A* Search on the graph
    
    Args:
        graph (Graph): The graph to search
        
    Returns:
        tuple: (goal_node, nodes_created, path)
            - goal_node: ID of destination reached
            - nodes_created: Number of nodes created during search
            - path: List of nodes in path from origin to goal
    """
    origin = graph.origin
    queue = []
    insertion_order = 0
    
    # Initial node: f = g + h, where g = 0 and h = heuristic
    h = graph.heuristic(origin)
    heapq.heappush(queue, (h, origin, insertion_order, [origin], 0))
    
    g_values = {origin: 0}
    
    nodes_created = 1
    
    while queue:
        _, node, _, path, g = heapq.heappop(queue)
        
        if graph.is_destination(node):
            return node, nodes_created, path
        
        for neighbor, cost in graph.get_neighbors(node):
            new_g = g + cost
   
            if neighbor not in g_values or new_g < g_values[neighbor]:
                g_values[neighbor] = new_g
                h = graph.heuristic(neighbor)
                f = new_g + h
                
                insertion_order += 1
                new_path = path + [neighbor]
                heapq.heappush(queue, (f, neighbor, insertion_order, new_path, new_g))
                nodes_created += 1
    
    # No path found
    return None, nodes_created, []

        
        
    