## Changelog

- 2025-04-09
  - Most of static methods now require an object instance of Graph to use. Only ady_to_dict() and dict_to_ady() remain static.
  - Added Dijkstra and Floyd-Warshall algorithm (last one is on preeliminar state)
- 2025-05-25
  - Changed the dict representation from alphabetic keys to numeric keys.
  - Improved Floyd-Warshal algorithm, added betweenness centrality method for all nodes of a graph.
  - Added method to export directly to .xlsx file.
