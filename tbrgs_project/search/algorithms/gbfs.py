"""
Greedy Best-First Search (GBFS)  
"""
import heapq

def search(graph):
    """
    Perform Greedy Best-First Search on the graph
    
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

    h = graph.heuristic(origin)
    heapq.heappush(queue, (h, insertion_order, origin, [origin]))

    visited = set()
    nodes_created = 1

    while queue:
        _, _, node, path = heapq.heappop(queue)

        if node in visited:
            continue
        
        visited.add(node)

        if graph.is_destination(node):
            return node, nodes_created, path
        
        neighbors = graph.get_neighbors(node)

        for neighbor, _ in neighbors:
            if neighbor not in visited:
                insertion_order += 1
                h = graph.heuristic(neighbor)
                new_path = path + [neighbor]
                heapq.heappush(queue, (h, insertion_order, neighbor, new_path))
                nodes_created += 1

    return None, nodes_created, []
