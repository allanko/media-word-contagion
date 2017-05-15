import numpy as np
import networkx as nx
import datetime
import csv

G = nx.read_gexf('network.gexf')

indices = dict([(name, i) for i, name in enumerate(G.nodes())])
max_interval = 16000

states = np.zeros((500, max_interval))

start_dt = datetime.datetime(2015, 4, 1, 0, 0, 0)

unfound = []
with open('stories_mentioning_altright.csv', 'rt') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='\"')
    for row in reader:
        if row[5][0].isalpha():
            continue
        name = row[1]
        dt = datetime.datetime.strptime(row[5], '%Y-%m-%d %H:%M:%S')
        minute_since_start = int((dt - start_dt).seconds / 3600)
        if name in indices:
            states[indices[name],minute_since_start:] = 1

A = nx.to_numpy_matrix(G)
np.save('adjacency', A.transpose())
np.save('states', states.transpose())
