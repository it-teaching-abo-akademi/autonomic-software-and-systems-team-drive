from collections import defaultdict


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

class PathGraph():
    def __init__(self):

        """
        self.edges: e.g. {'X': ['A', 'B', 'C', 'E'], ...}
        self.weights: e.g. {('X', 'A'): 7, ('X', 'B'): 2, ...}
        """
        
        self.edges = defaultdict(list)
        self.weights = {}
        self.data = {}
    
    def add_edge(self, from_node, to_node, weight, from_data, to_data):
    
        self.edges[from_node].append(to_node)
        self.weights[(from_node, to_node)] = weight
        self.data[(from_node,to_node)] = (from_data,to_data)

    def dijsktra(self, initial, end):
        graph = self
        shortest_paths = {initial: (None, 0)}
        current_node = initial
        visited = set()
        
        while current_node != end:
            visited.add(current_node)
            destinations = graph.edges[current_node]
            weight_to_current_node = shortest_paths[current_node][1]

            for next_node in destinations:
                weight = graph.weights[(current_node, next_node)] + weight_to_current_node
                if next_node not in shortest_paths:
                    shortest_paths[next_node] = (current_node, weight)
                else:
                    current_shortest_weight = shortest_paths[next_node][1]
                    if current_shortest_weight > weight:
                        shortest_paths[next_node] = (current_node, weight)
            
            next_destinations = {node: shortest_paths[node] for node in shortest_paths if node not in visited}
            if not next_destinations:
                return []

            current_node = min(next_destinations, key=lambda k: next_destinations[k][1])
        
        # Work back through destinations in shortest path
        path = []
        while current_node is not None:
            path.append(current_node)
            next_node = shortest_paths[current_node][0]
            current_node = next_node
        # Reverse path
        path = path[::-1]

        # Convert to Waypoints
        if(len(path)<2):
            return []
        waypoints = [graph.data[(path[i-1], path[i])] for i in range(1,len(path))]
        return waypoints