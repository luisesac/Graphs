import random
import time
from collections import defaultdict, deque
import heapq
import numpy as np
import pandas as pd

INF = float('inf')

class Graph:
    def __init__(self, graph):
        if isinstance(graph, dict):
            self.g = graph
            self.nodes = graph.keys()
            self.ady = self.dict_to_ady()
        elif isinstance(graph, list):
            self.g = (Graph.ady_to_dict(graph))
            self.nodes =  range(len(graph))
            self.ady = graph
        
    def edges(self):
        edges_ = []
        for node in self.nodes:
            for neighbors in self.g[node].keys():
                if node != neighbors:
                    edges_.append((node, neighbors))
        return edges_
    
    def __str__(self):
        return "\n".join(str(row) for row in self.ady)
    
    def random(size = 5, directed = False, weighted = False, maxWeight=10):
        d = [[0 for _ in range(size)] for _ in range(size)]
        for i in range(size):
            print("="*5, f"Generating random Graph ({i})", "="*5)
            for j in range(size):
                add = random.choice([0, 1])
                if i == j:
                    w = random.randint(1, maxWeight)*add if directed else 0
                    d[i][j] = w
                if not directed and j < i:
                    continue
                else:
                    if add: 
                        w = random.randint(1, maxWeight) if weighted else 1
                        d[i][j] = w
                        if not directed:
                            d[j][i] = w
        return Graph(d)

    def degree(self):
        deg = []
        for i in self.ady:
            d = 0
            for j in i:
                if j > 0:
                    d+=1
            deg.append(d)
        return deg
    
    #   Return the dictionary-defined graph from an adjacency matrix 
    def ady_to_dict(g)->dict:
        nodes = range(len(g))
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
                    weights.append(float('inf'))
            graph_dict[nodes[i]] = (dict(zip(neighbors, weights)))
        return graph_dict
    
    #   Return the dictionary-like representation of an adjacency matrix of a Graph
    def dict_to_ady(self)->list:
        nodes = list(self.g.keys())
        num_nodes = len(nodes)
        graph_ady = [[float('inf') for _ in range(num_nodes)] for _ in range(num_nodes)]
        for i in range(num_nodes):
            node = self.g[nodes[i]]
            neighbors = list(self.g[nodes[i]].keys())
            for j in range(num_nodes):
                if i == j:
                    graph_ady[i][j] = 0
                elif nodes[j] in neighbors:
                    graph_ady[i][j] = self.g[nodes[i]][nodes[j]]
        return graph_ady
    
    def path_cost(self, path:list):
        cost = 0
        for i in range(len(path)-1):
            if self.ady[path[i]] != 0 and self.ady[path[i+1]] != 0:
                cost += self.ady[path[i]][path[i+1]]
            else:
                print("There is no connection on this path.")
                return 0
        return cost
    
    def find_all_paths(self):
        adj_matrix = self.ady
        num_nodes = len(adj_matrix)
        all_paths = []

        def dfs(current, path, visited):
            visited[current] = True
            path.append(current)

            for neighbor, connected in enumerate(adj_matrix[current]):
                print("="*5, "Finding all walks", "="*5)
                if connected:
                    if not visited[neighbor]:
                        dfs(neighbor, path.copy(), visited.copy())
            
            if len(path) > 1:
                all_paths.append(path)

        for start_node in range(num_nodes):
            visited = [False] * num_nodes
            dfs(start_node, [], visited)

        return all_paths

    def dijkstra(self, source:str):
        #   Dictionary graph representation
        nodes = []
        distances = {}
        previous = {}
        visited = {}
        for i in list(self.g.keys()):
            nodes.append(i)
            distances[i] = float('inf')
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
            for v in self.g.get(u):
                if not visited[v] and distances[v] > distances[u] + self.g[u][v]:
                    distances[v] = distances[u] + self.g[u][v]
                    previous[v] = u
                    queue.append((distances[v], v))
        return distances, previous

    def fw(self):
        S = []
        D = [[0 for _ in i] for i in self.ady]
        P = [[0 for _ in i] for i in self.ady]
        for i in range(len(self.nodes)):
            for j in range(len(self.nodes)):
                if i == j:
                    D[i][j] = 0
                    P[i][j] = None
                elif tuple((self.nodes[i], self.nodes[j])) in self.edges():
                    D[i][j] = self.g[self.nodes[i]][self.nodes[j]]
                    P[i][j] = j
                else:
                    D[i][j] = INF
                    P[i][i] = None
        edges = self.edges()
        while S != self.edges():
            pivot = edges.pop(0)
            for w in range(len(self.nodes)):
                for v in range(len(self.nodes)):
                    for u in range(len(self.nodes)):
                        if D[u][w] + D[w][v] < D[u][v]:
                            D[u][v] = D[u][w] + D[w][v]
                            P[u][v] = P[u][w]
            S.append(pivot)
        return D, P
    
    #   Returns (2*|n|)/(|e|*(|e|-1)). n:= nodes, e:= vertices
    def density(self):
        d = self.ady_to_dict(self.g)
        nodes = [i[0] for i in d.items()]
        edges = [{tuple(sorted([node1, node2])) for node1 in nodes for node2 in d[node1]}]
        len_edges = len(edges)
        len_nodes = len(nodes)
        return (2*len_edges)/(len_nodes*(len_nodes-1))
    
    #   Returns the list of the shortest paths between each node and the shortest of them.
    def excentricity(self):
        for i in self.nodes:
            d = self.dijkstra(i)[0]
            mins = []
            min_val = float('inf')
            for k in d.keys():
                if d[k] < min_val and d[k] > 0:
                    min_val = d[k]
            mins.append(min_val)
        return (mins)

    def rad(self):
        m = min(self.excentricity())
        return m
    
    def diam(self):
        m = max(self.excentricity())
        return m
    
    """IN PROGRESS"""
    def fw_asp(self):
        dist = self.g
        #distance = list(map(lambda i: list(map(lambda j: j, i)), dist))
        n_vertices = len(dist)
        for k in range(n_vertices):
            for i in range(n_vertices):
               for j in range(n_vertices):
                   if dist[i][k] != float('inf') and dist[k][j] != float('inf'):
                       dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
        return dist
    
    def centrality_closeness(self):
        d = self.g
        dist = [1/sum(v) for v in d]
        return dist

    def graph_centrality(self):
        d = self.g
        dist = [1/max(v) for v in d]
        return dist
    
    #   CREATE FOR ALL NODES INSTEAD OF ONE!!!!
    def centrality_betweenness(self, v):
        paths = self.find_all_paths()
        walks = []
        unique_walks = []
        for path in paths:
            print("="*5, "Checking paths", "="*5)
            #   Check that the v node is within the path but not in the origin or end of it.
            if v not in [path[0], path[-1]] and (path[-1], path[0]) not in unique_walks:
                walks.append(((path[0], path[-1]),path, self.path_cost(path)))
                unique_walks.append((path[0], path[-1]))
        unique_walks = list(set(unique_walks))
        walks.sort()
        unique_walks.sort()
    
        walk_pairs = {i : [] for i in unique_walks}
        
        #   pairs for centrality betweenness. If the walk is a minimun distance, it is counted to the metric.
        for walk in walks:
            print("="*5, "Checking walks", "="*5)
            if walk[2] == self.dijkstra(walk[0][0])[0][walk[0][1]]:
                walk_pairs[walk[0]].append(walk[1])  

        betweennes_centrality = {i : 0 for i in unique_walks}
        
        for i in walk_pairs.keys():
            print("="*5, "Checking pairs", "="*5)
            d = len(walk_pairs[i])
            n = 0
            for j in walk_pairs[i]:
                if v in j:
                    n += 1
            betweennes_centrality[i] = n/d
        bc = {}
        for i in walk_pairs.keys():
            print("="*5, "Checking pairs(2)", "="*5)
            if betweennes_centrality[i] > 0:
                bc[i] = betweennes_centrality[i]
        return bc

    def brandes_betweenness_centrality(self, directed=False, weighted=False):
        """
        Calcula la centralidad de intermediación usando el algoritmo de Brandes.
        - directed: Indica si el grafo es dirigido
        - weighted: Indica si el grafo es ponderado
        """
        n = len(self.ady)
        betweenness = defaultdict(float)
        
        for s in range(n):
            # Estructuras de datos para el recorrido
            S = []
            P = defaultdict(list)
            sigma = [0] * n
            sigma[s] = 1
            d = [-1] * n
            d[s] = 0
            
            if weighted:
                # Cola de prioridad para Dijkstra
                heap = []
                heapq.heappush(heap, (0, s))
            else:
                # Cola normal para BFS
                queue = deque([s])

            while (weighted and heap) or (not weighted and queue):
                if weighted:
                    dist_v, v = heapq.heappop(heap)
                    if d[v] < dist_v:  # Una mejor distancia
                        continue
                else:
                    v = queue.popleft()
                
                S.append(v)
                
                for w in range(n):
                    if self.ady[v][w] == 0 or v == w:  # No hay conexión o es un bucle
                        continue
                    
                    weight = self.ady[v][w]
                    
                    # Descubrimiento de w
                    if d[w] == -1:
                        if weighted:
                            d[w] = d[v] + weight
                            heapq.heappush(heap, (d[w], w))
                        else:
                            d[w] = d[v] + 1
                            queue.append(w)
                    
                    # Existe un camino más corto a través de v
                    if d[w] == d[v] + weight:
                        sigma[w] += sigma[v]
                        P[w].append(v)
            
            # Acumulación de dependencias
            delta = defaultdict(float)
            while S:
                w = S.pop()
                for v in P[w]:
                    if sigma[w] != 0: 
                        delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
                if w != s:
                    betweenness[w] += delta[w]
        
        # Ajuste para grafos no dirigidos
        if not directed:
            for v in betweenness:
                betweenness[v] /= 2
        
        # Convertir a lista ordenada
        return [betweenness[v] for v in range(n)]
    
    def to_xlsx(self):
        rows = []
        graph = self.ady
        for i in range(len(graph)):
            for j in range(len(graph)):
                if 0 < graph[i][j] < float('inf'):
                    rows.append({
                    "source_node": i,
                    "interactive_value": graph[i][j],
                    "target_node": j
                })
        df = pd.DataFrame(rows)
        return df.to_excel("output.xlsx", index=False)


#   Use example:
def main():
    now = time.time()
    
    ady = [
        [0, 1, 1, 1, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 0],
        [1, 1, 0, 1, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 1, 1, 0]
    ]

    g = Graph(ady)

    rg = Graph.random(5, directed=True, weighted=True)

    print(rg)

    rg.to_xlsx()

    end = time.time()

    print(1000*(end - now), "ms")
    return

if __name__ == "__main__":
    main()