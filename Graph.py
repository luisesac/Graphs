import numpy as np
import string

class Graph:
    """
    The default representation of a graph in this class is the dictionary type, where each key (node) has it's corresponding match
    and weigth. Can be used to represent directed or non-directed graphs with or without weights (1).
    That being said, it can recieve a adjacency matrix to create the object, yet the representation will be a dictionary.

    The goal of this class is to be a tool for the optative course "Redes complejas" for the career "Inteligencia Artificial" of the "Universidad Autónoma 
    del Estado de Morelos" (UAEM) imparted by Dr. Elizabeth Santiago de Ángel.

    While this class does not need an instance, it can provide tools to operate over different representation of Graphs that can be scalable and will
    be useful throughout the course.
    """

    def __init__(self):
        raise Exception("Cannot create an instance of this class because it only contains static methods!")

    #   Return the dictionary-defined graph from an adjacency matrix 
    @staticmethod
    def ady_to_dict(graph:list)->dict:
        nodes = list(string.ascii_lowercase[:len(graph)])
        graph_dict = dict(zip(nodes, [None for _ in range(len(nodes))]))
        for i in range(len(nodes)):
            neighbors = []
            weights = []
            for j in range(len(nodes)):
                if graph[i][j] > 0:
                    neighbors.append(nodes[j])
                    weights.append(graph[i][j])
            graph_dict[nodes[i]] = (dict(zip(neighbors, weights)))
        return graph_dict

    @staticmethod
    def dijkstra(graph:dict, source:str):
        #   Dictionary graph representation
        if isinstance(graph, dict):
            nodes = []
            distances = {}
            previous = {}
            visited = {}
            for i in list(graph.keys()):
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
                #   Tuple is not needed so we only take the name of the node for better legibility.
                u = u[1]
                visited[u] = True
                for v in graph.get(u):
                    if not visited[v] and distances[v] > distances[u] + graph[u][v]:
                        distances[v] = distances[u] + graph[u][v]
                        previous[v] = u
                        queue.append((distances[v], v))
            return distances, previous
        #   Adyacence matrix
        elif isinstance(graph, list):
            return Graph.dijkstra(Graph.ady_to_dict(graph), source)
        else: raise Exception("Graph must be a dictionary like {source: {destination: weight}}, or an adjacency matrix.")

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

    ady = [
        [0, 3, 0, 0, 0, 6],
        [3, 0, 5, 0, 5, 1],
        [0, 5, 0, 4, 3, 0],
        [0, 0, 4, 0, 3, 0],
        [0, 5, 3, 3, 0, 2],
        [6, 1, 0, 0, 2, 0]
    ]

    print(Graph.ady_to_dict(ady))
    print(g)
    nodos = g.keys()
    source = 'c'
    print('source:', source)
    print(Graph.dijkstra(ady, source))


if __name__ == "__main__":
    main()