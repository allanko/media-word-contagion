import networkx as nx
import numpy as np

G = nx.read_gexf('test.gefx', relabel=True)

print(G.nodes())

# some graph G
A = nx.to_numpy_matrix(G)
print(nx.diameter(G))
print(np.max(A))
print(A.shape)
print(A.sum(axis=1)[:, np.newaxis].shape)
# A_ij is the probability that i takes the term from j
