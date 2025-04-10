import numpy as np
import string

class Graph:
    """
    The goal of this class is to be a tool for the optative course "Redes complejas" for the career "Inteligencia Artificial" of the "Universidad Autónoma 
    del Estado de Morelos" (UAEM) imparted by Dr. Elizabeth Santiago de Ángel.
    """

    def __init__(self, graph):
        if isinstance(graph, dict):
            self.g = (Graph.dict_to_ady(graph))
        elif isinstance(graph, list):
            self.g = Graph.dict_to_ady(Graph.ady_to_dict(graph))

    def __str__(self):
        return "\n".join(str(row) for row in self.g)

    #   Return the dictionary-defined graph from an adjacency matrix 
    @staticmethod
    def ady_to_dict(g)->dict:
        nodes = list(string.ascii_lowercase[:len(g)])
        graph_dict = dict(zip(nodes, [None for _ in range(len(nodes))]))
        for i in range(len(nodes)):
            neighbors = []
            weights = []
            for j in range(len(nodes)):
                neighbors.append(nodes[j])
                if i == j:
                    weights.append(0)
                elif g[i][j] > 0:
                    weights.append(g[i][j])
                else:
                    weights.append(np.inf)
            graph_dict[nodes[i]] = (dict(zip(neighbors, weights)))
        return graph_dict
    
    #   Return the dictionary-like representation of an adjacency matrix of a Graph
    @staticmethod
    def dict_to_ady(g)->list:
        nodes = list(g.keys())
        num_nodes = len(nodes)
        graph_ady = [[np.inf for _ in range(num_nodes)] for _ in range(num_nodes)]
        for i in range(num_nodes):
            node = g[nodes[i]]
            neighbors = list(g[nodes[i]].keys())
            for j in range(num_nodes):
                if i == j:
                    graph_ady[i][j] = 0
                elif nodes[j] in neighbors:
                    graph_ady[i][j] = g[nodes[i]][nodes[j]]
        return graph_ady

    def dijkstra(g, source:str):
        #   Dictionary graph representation
        nodes = []
        distances = {}
        previous = {}
        visited = {}
        for i in list(g.keys()):
            nodes.append(i)
            distances[i] = np.inf
            previous[i] = None
            visited[i] = False
        distances[source] = 0
        previous[source] = source
        queue = [(distances[source], source)]

        while(queue):
            u = min(queue)
            queue.remove(u)

            #   Full tuple is not needed so we only take the name of the node for better legibility.
            u = u[1]
            visited[u] = True
            for v in g.get(u):
                if not visited[v] and distances[v] > distances[u] + g[u][v]:
                    distances[v] = distances[u] + g[u][v]
                    previous[v] = u
                    queue.append((distances[v], v))
        return distances, previous

    #   Returns (2*|n|)/(|e|*(|e|-1)). n:= nodes, e:= vertices
    def density(self):
        d = self.ady_to_dict(self.g)
        nodes = [i[0] for i in d.items()]
        edges = [{tuple(sorted([node1, node2])) for node1 in nodes for node2 in d[node1]}]
        len_edges = len(edges)
        len_nodes = len(nodes)
        return (2*len_edges)/(len_nodes*(len_nodes-1))
    
    """IN PROGRESS"""
    def fw_asp(self):
        distance = list(map(lambda i: list(map(lambda j: j, i)), self.g))
        n_vertices = len(self.g)
        for k in range(n_vertices):
            for i in range(n_vertices):
                for j in range(n_vertices):
                    distance[i][j] = min(distance[i][j], distance[i][k] + distance[k][j])
        return distance

#   Use example:
def main():
    g = {
        "a" : {"b" : 3, "f" : 6},
        "b" : {"a" : 3, "c" : 5, "e" : 5, "f" : 1},
        "c" : {"b" : 5, "d" : 4, "e" : 3},
        "d" : {"c" : 4, "e" : 1},
        "e" : {"b" : 5, "c" : 3, "d" : 1, "f" : 2},
        "f" : {"a" : 6, "b" : 1, "e" : 2}
    }

    g1 = {
        "a" : {"e" : 2},
        "b" : {"a" : 4},
        "c" : {"d" : 3},
        "d" : {"b" : 6},
        "e" : {"b" : 1, "d" : 12}
    }

    ady = [
        [0, 3, 0, 0, 0, 6],
        [3, 0, 5, 0, 5, 1],
        [0, 5, 0, 4, 3, 0],
        [0, 0, 4, 0, 3, 0],
        [0, 5, 3, 3, 0, 2],
        [6, 1, 0, 0, 2, 0]
    ]
    g = Graph(g)
    print(Graph(g.fw_asp()))
    print(g.density())
    #print(Graph.fw_asp(Graph.ady_to_dict(ady)))
"""
    print(Graph.ady_to_dict(ady))
    print(g)
    nodos = g.keys()
    source = 'c'
    print('source:', source)
    print(Graph.dijkstra(ady, source))
    print(Graph.density(g))
    """

if __name__ == "__main__":
    main()