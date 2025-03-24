"""
Graph data structure for route finding problem
"""

class Graph:
    def __init__(self, nodes, edges, origin, destinations):
        """
        Initialize a graph for the route finding problem
        
        Args:
            nodes (dict): Dictionary mapping node IDs to (x,y) coordinates
            edges (dict): Dictionary mapping (from_node, to_node) to edge cost
            origin (int): Starting node ID
            destinations (list): List of destination node IDs
        """
        self.nodes = nodes
        self.edges = edges
        self.origin = origin
        self.destinations = destinations

    def get_neighbors(self, node_id):
        """
        Get all neighboring nodes that can be reached from the given node
        Returns list of (neighbor_id, cost) tuples, sorted by node ID
        """
        neighbors = []
        for (from_node, to_node), cost in self.edges.items(): 
            if from_node == node_id:
                neighbors.append((to_node, cost))

        # Sort neighbors by node ID (tie-breaking rule)
        return sorted(neighbors, key=lambda x: x[0])

    def is_destination(self, node_id):
        """Check if a node is a destination node"""
        return node_id in self.destinations

    def get_coordinates(self, node_id):
        """Get the coordinates of a node"""
        return self.nodes[node_id]
    
    def heuristic(self, node_id, destination_id=None):
        """
        Calculate heuristic value (straight-line distance) from node to destination
        If no specific destination given, finds minimum distance to any destination
        """
        if destination_id is not None:
            x1, y1 = self.nodes[node_id]
            x2, y2 = self.nodes[destination_id] 
            return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 # Euclidean distance formula
        else:
            min_distance = float('inf')
            for destination in self.destinations:
                distance = self.heuristic(node_id, destination)
                if distance < min_distance:
                    min_distance = distance
            return min_distance
