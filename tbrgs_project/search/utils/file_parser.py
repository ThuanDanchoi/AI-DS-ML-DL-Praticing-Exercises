"""
File parser utility for route finding problem
"""

def parse_problem_file(filename):
    """
    Parse a problem file and extract nodes, edges, origin, and destinations
    
    Args:
        filename (str): Path to the problem file
        
    Returns:
        tuple: (nodes, edges, origin, destinations)
            - nodes: dict mapping node IDs to (x,y) coordinates
            - edges: dict mapping (from_node, to_node) to cost
            - origin: starting node ID
            - destinations: list of destination node IDs
    """
    nodes = {}
    edges = {}
    origin = None
    destinations = []
    
    current_section = None
    
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Check for section headers
            if line == "Nodes:":
                current_section = "nodes"
                continue
            elif line == "Edges:":
                current_section = "edges"
                continue
            elif line == "Origin:":
                current_section = "origin"
                continue
            elif line == "Destinations:":
                current_section = "destinations"
                continue
            
            # Parse content based on current section
            if current_section == "nodes":
                # Parse node 
                node_id, coords = line.split(":")
                node_id = int(node_id.strip())
                
                # Extract x,y  
                coords = coords.strip()
                coords = coords.strip("()")
                x, y = map(int, coords.split(","))
                
                nodes[node_id] = (x, y)
                
            elif current_section == "edges":
                # Parse edge 
                edge, cost = line.split(":")
                edge = edge.strip()
                edge = edge.strip("()")
                from_node, to_node = map(int, edge.split(","))
                cost = int(cost.strip())
                
                edges[(from_node, to_node)] = cost
                
            elif current_section == "origin":
                # Parse origin 
                origin = int(line.strip())
                
            elif current_section == "destinations":
                # Parse destinations
                dest_list = line.strip().split(";")
                destinations = [int(d.strip()) for d in dest_list]
    
    return nodes, edges, origin, destinations