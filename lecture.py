import shutil

def line_break():
    terminal_width = shutil.get_terminal_size().columns
    line = '=' * terminal_width
    print(line)


##########
# Graphs #
##########

# Used to represent relationships between pairs of objects that consist vertices (nodes) and set of edges that connect the vertices

# Types of Graphs
# Directed Graph - 

class DirectedGraph:
    def __init__(self):
        self.graph = {}

    def add_edge(self, u, v):
        if u not in self.graph:
            self.graph[u] = []
        self.graph[u].append(v)

    def print_graph(self):
        for vertex, edges in self.graph.items():
            print(f"Vertex {vertex} -> {edges}")

dg = DirectedGraph()
dg.add_edge(0, 1)
dg.add_edge(0, 2)
dg.add_edge(1, 2)
dg.add_edge(2, 3)
dg.add_edge(3, 4)

dg.print_graph()


line_break()

# Undirected Graphs - Two Way Streets
class UndirectedGraph:
    def __init__(self):
        self.graph = {}

    def add_edge(self, u, v):
        if u not in self.graph:
            self.graph[u] = []
        if v not in self.graph:
            self.graph[v] = []
        self.graph[u].append(v)
        self.graph[v].append(u)

    def print_graph(self):
        for vertex, edges in self.graph.items():
            print(f"Vertex {vertex}: {edges}")

ug = UndirectedGraph()
ug.add_edge(0, 1)
ug.add_edge(0, 2)
ug.add_edge(1, 2)
ug.add_edge(2, 3)
ug.add_edge(3, 4)

ug.print_graph()


# Adjacency Matrices
# Imagine you have a map with all the locations (vertices) you want to visit on your road trip. An adjacency matrix 
# is like a table where each row and column represent a location, and the entries indicate whether there's a 
# direct road (edge) between those locations. If there's a road connecting two locations, the corresponding entry is 1; otherwise, it's 0.
# Example adjacency matrix for a graph with 3 vertices

adj_matrix = [
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 0],
]


# Adjacency Lists
# An adjacency list is a more compact representation of a graph. It lists each vertex and its neighboring
# vertices. This representation is useful for sparse graphs where there are fewer connections between vertices.
# Example adjacency list for a graph with 4 vertices

adj_list = {
    1: [2, 3],
    2: [1, 3, 4],
    3: [1, 2],
    4: [2]
}

# Terminology

# Degree of a Vertex
# Number of edges connected to that vertex

# Path
# Sequence of vertices where each adjacent pair of vertices is connected by an edge aka the route from one vertex to another

# Cycle
# A path that starts and ends at the same vertex, traversing through a sequence of distinct vertices and edges without repetition

line_break()

class Graph:
    def __init__(self):
        self.vertices = {}

    def add_vertex(self, vertex):
        # If the vertex is not in the dictionary of vertices
        if vertex not in self.vertices:
            # Add the vertex to the dictionary with an empty list as the value
            self.vertices[vertex] = [] # Will be a list of neighboring vertices

    def add_edge(self, vertex1, vertex2):
        # Make sure that both vertices are in the graph
        if vertex1 in self.vertices and vertex2 in self.vertices:
            # Add an edge going from vertex1 to vertex2
            self.vertices[vertex1].append(vertex2)
            # Same edge going frrom vertex2 to vertex1
            self.vertices[vertex2].append(vertex1) # For undirected graph

    def display(self):
        # Loop through the vertices dictionary
        for vertex, neighbors in self.vertices.items():
            print(f"Vertex {vertex} -> {neighbors}")

    def has_path(self, start, end, visited=None):
        # Initialize the visited set if not provided
        if visited is None:
            visited = set()
        # Mark the current vertex as visited
        visited.add(start)
        # Best case: If start and end are the same, return True
        if start == end:
            return True
        # Traverse neighbors of the current vertex
        for neighbor in self.vertices[start]:
            # If the neighbor has not been visited, recursively check if there is a path from it to the end
            if neighbor not in visited:
                if self.has_path(neighbor, end, visited):
                    return True
        # If no path is found from any neighbor to the end vertex, return False
        return False


my_graph = Graph()
my_graph.add_vertex(1)
my_graph.add_vertex(2)
my_graph.add_vertex(3)
my_graph.add_vertex(4)
my_graph.add_vertex(5)

my_graph.add_edge(1, 2)
my_graph.add_edge(2, 3)
my_graph.add_edge(3, 4)
my_graph.add_edge(4, 1)

my_graph.display()

print("Path exists between 1 and 3:", my_graph.has_path(1, 3))
print("Path exists between 1 and 5:", my_graph.has_path(1, 5))


line_break()
line_break()

########################
# Dijkstra's Algorithm #
########################

# Dijkstra's Algorithm is a fundamental graph algorithm used to find the shortest path between two nodes in a weighted graph. It works well for graphs with non-negative edge weights.

# Imagine you're navigating through a city using Google Maps, and you want to find the shortest route from your current location to a destination. Dijkstra's Algorithm is like having a smart GPS that calculates the most efficient path for you, considering the distances between intersections and the traffic conditions.

# Here's how Dijkstra's Algorithm works:

# 1. **Initialization**: Start by assigning a tentative distance value to every node. Set the initial node's distance to 0 and all other nodes' distances to infinity.
# 2. **Visit Neighbors**: Explore all the neighboring nodes of the current node and update their tentative distances if a shorter path is found.
# 3. **Select Next Node**: Choose the node with the smallest tentative distance as the next current node.
# 4. **Repeat**: Repeat steps 2 and 3 until all nodes have been visited.

# The result is a shortest path tree from the source node to all other nodes in the graph.

import heapq

class Graph:
    def __init__(self):
        self.vertices = {}

    def add_vertex(self, vertex):
        # If the vertex is not in the dictionary of vertices
        if vertex not in self.vertices:
            # Add the vertex to the dictionary with an empty list as the value
            self.vertices[vertex] = {} # Will be a dictionary of neighboring vertices as key and weight as value

    def add_edge(self, vertex1, vertex2, weight):
        # Make sure that both vertices are in the graph
        if vertex1 in self.vertices and vertex2 in self.vertices:
            # Add an edge going from vertex1 to vertex2 with weight
            self.vertices[vertex1][vertex2] = weight
            # Same edge going frrom vertex2 to vertex1 with weight
            self.vertices[vertex2][vertex1] = weight # For undirected graph

    def dijkstra(self, start):
        # Initialize distances with infinty for all vertices except the start
        distances = {vertex: float('inf') for vertex in self.vertices}
        distances[start] = 0

        # Priority queue of (distance, vertex) pairs
        pq = [(0, start)]

        while pq:
            # Get the vertex with the smallest distance from the priority queue
            current_distance, current_vertex = heapq.heappop(pq)

            # If the current_distance is greater than the distance already calculated for this vertex, skip this step
            if current_distance > distances[current_vertex]:
                continue

            # Explore the neighbors of the current_vertex
            for neighbor, weight in self.vertices[current_vertex].items():
                # Calculate the distance to the neighbor through the current vertex
                distance = current_distance + weight

                # If this path to the neighbor is shorter than any found previously, update the distance
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    # Add the nieghbor to the priority queue with its updated distance
                    heapq.heappush(pq, (distance, neighbor))
        # after the while loop, return distances
        return distances

graph1 = Graph()
graph1.add_vertex('A')
graph1.add_vertex('B')
graph1.add_vertex('C')
graph1.add_vertex('D')
graph1.add_edge('A', 'B', 4)
graph1.add_edge('B', 'C', 3)
graph1.add_edge('C', 'D', 2)
graph1.add_edge('C', 'A', 5)

distances = graph1.dijkstra('C')
print('Shortest distances from vertex C:', distances)



line_break()
line_break()

class NavigationSystem:
    def __init__(self):
        self.vertices = {}

    def add_landmark(self, landmark):
        if landmark not in self.vertices:
            self.vertices[landmark] = {}

    def add_route(self, landmark1, landmark2, distance):
        if landmark1 in self.vertices and landmark2 in self.vertices:
            self.vertices[landmark1][landmark2] = distance
            self.vertices[landmark2][landmark1] = distance # Undirected Graph

    def dijkstra(self, start):
        distances = {landmark: float('inf') for landmark in self.vertices}
        distances[start] = 0
        paths = {landmark: [] for landmark in self.vertices}

        # Priority queue of (distance, vertex) pairs
        pq = [(0, start)]

        while pq:
            # Get the vertex with the smallest distance from the priority queue
            current_distance, current_landmark = heapq.heappop(pq)

            # If the current_distance is greater than the distance already calculated for this vertex, skip this step
            if current_distance > distances[current_landmark]:
                continue

            # Explore the neighbors of the current_landmark
            for neighbor, weight in self.vertices[current_landmark].items():
                # Calculate the distance to the neighbor through the current vertex
                distance = current_distance + weight

                # If this path to the neighbor is shorter than any found previously, update the distance
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    paths[neighbor] = paths[current_landmark] + [current_landmark] # Update path
                    # Add the nieghbor to the priority queue with its updated distance
                    heapq.heappush(pq, (distance, neighbor))
        # after the while loop, return distances
        return distances, paths


# Example Usage
gps = NavigationSystem()

gps.add_landmark('Home')
gps.add_landmark('Office')
gps.add_landmark('Gym')
gps.add_landmark('Store')
gps.add_landmark('Pub')

gps.add_route('Home', 'Office', 2)
gps.add_route('Home', 'Gym', 5)
gps.add_route('Office', 'Gym', 4)
gps.add_route('Office', 'Store', 1)
gps.add_route('Gym', 'Store', 6)
gps.add_route('Store', 'Pub', 3)


home_distances = gps.dijkstra('Home')
print('Shortest distances from Home:', home_distances)

line_break()

store_distances = gps.dijkstra('Store')
print('Shortest distances from the Store:', store_distances)

