"""
Embedding Generator with Analysis

Generates the custom embedding for all nodes in the combined ontology graph, a plot 
comparing the cosine similarity between custom embeddings and the minimum path length
between nodes in the graph, and a statistical analysis of the relationship shown in the
plot.
"""

#Install rdflib
!pip install rdflib

#Import packages
import rdflib
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


def component(source:str,target:str,shortest_paths:dict) -> int:
  """
  Computes the j-th component of the vector representation of the i-th node
  in the subClassOf graph according to the custom embedding introduced in the
  paper.

  Args:

    - source: the first node.
    - target: the second node
    - shortest_paths: a dictionary listing the shortest distance between
    all nodes in the graph.

  Returns:

    - The j-th component of the vector representation of the i-th node
      in the subClassOf graph according to the custom embedding introduced in
      the paper.
  """
  if shortest_paths[source][target] == float('inf'):
    return 0
  if shortest_paths[source][target] == 0:
    return 1
  else:
    return 1/shortest_paths[source][target]

def cosine_similarity_vectors(v1:list, v2:list) -> float:
  """Computes the cosine similarity between two vectors."""
  v1 = np.array(v1)
  v2 = np.array(v2)
  dot_product = np.dot(v1, v2)
  magnitude_v1 = np.linalg.norm(v1)
  magnitude_v2 = np.linalg.norm(v2)
  if magnitude_v1 == 0 or magnitude_v2 == 0:
    return 0
  return dot_product / (magnitude_v1 * magnitude_v2)

def load_graph_from_rdf(rdf_path:str) -> nx.Graph:
  """
  Loads an RDF file and returns a networkx graph object encoding the subClassOf
  relationships between classes.

  Args:

    - rdf_path: the path to the RDF file.

  Returns:

    - A networkx graph object.
  """
  g = rdflib.Graph()
  g.parse(rdf_path)

  nx_graph = nx.Graph()
  subclass_pred = rdflib.RDFS.subClassOf

  for subj, pred, obj in g.triples((None, subclass_pred, None)):
      nx_graph.add_edge(str(subj), str(obj))

  return nx_graph

def compute_all_min_path_lengths_with_unreachable(graph:nx.Graph) -> dict:
  """
  Generates a dictionary encoding the shortest path length between any two nodes
  in a graph.

  Args:

    - graph: a networkx graph object.

  Returns:
    - A dictionary encoding the shortest path length between any two nodes in
    the graph.
  """
  nodes = list(graph.nodes())
  shortest_paths = dict(nx.all_pairs_shortest_path_length(graph))
  min_path_lengths = {}
  for u in nodes:
      min_path_lengths[u] = {}
      for v in nodes:
          if u == v:
              min_path_lengths[u][v] = 0
          elif v in shortest_paths.get(u, {}):
              min_path_lengths[u][v] = shortest_paths[u][v]
          else:
              min_path_lengths[u][v] = float('inf')
  return min_path_lengths

#Load the combined ontology rdf as a graph object.
graph = load_graph_from_rdf("/rdsf/combined_ontology.rdf")

#Compute the shortest path between all nodes in the graph.
shortest_paths = compute_all_min_path_lengths_with_unreachable(graph)

#Build a dictionary containing the custom embedding for each node in the graph.
embedding_dict = {}
for node in list(graph.nodes()):
  embedding_dict[node] = [component(node,n,shortest_paths) for n in
                          list(graph.nodes())]

#Build a dictionary containing: 1) the cosine similarity between the custom
#embeddings for each for each pair of nodes, and 2) the minimum path length in
#the subClassOf graph between each pair of nodes. Takes ~90 seconds on CPU.
data = []
for node in tqdm(list(graph.nodes())):
  for n in list(graph.nodes()):
    data.append({
        "pair": (node,n),
        "similarity": cosine_similarity_vectors(embedding_dict[node],
                                                embedding_dict[n]),
        "path_length": shortest_paths[node][n]
    })


#Plot the relationship between path length and cosine similarity.
df = pd.DataFrame(data)
df['path_length_jitter'] = df['path_length'] + np.random.normal(0, 0.1, len(df))
plt.figure(figsize=(10, 6))
plt.scatter(df['path_length_jitter'], df['similarity'], alpha=0.1)
plt.xlabel('Minimum Path Length')
plt.ylabel('Cosine Similarity')
plt.show()


#Calculate the correlation between path length and cosine similarity.
correlation, p_value = pearsonr(df['similarity'], df['path_length'])
print(f"Correlation: {correlation}")
print(f"P-value: {p_value}")
