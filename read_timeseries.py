import numpy as np
import networkx as nx
import datetime
import csv

G = nx.read_gexf('network.gexf')

indices = dict([(name, i) for i, name in enumerate(G.nodes())])
max_interval = 20000

states = np.zeros((500, max_interval))

start_dt = datetime.datetime(2015, 4, 30, 0, 0, 0)

unfound = []
node_list = []
with open('stories_mentioning_altright.csv', 'rt') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='\"')
    for row in reader:
        if row[5][0].isalpha():
            continue
        name = row[1]
        dt = datetime.datetime.strptime(row[5], '%Y-%m-%d %H:%M:%S')
        diff = dt - start_dt
        #print(dt)
        hour_since_start = diff.days * 24 + int(diff.seconds / 3600)
        #print(hour_since_start)
        if name in indices and hour_since_start >= 0:
            node_list.append(name)
            if hour_since_start > 20000:
                print(hour_since_start, dt, row)
            states[indices[name],hour_since_start:] = 1

A = nx.to_numpy_matrix(G)
np.save('adjacency', A.transpose())
np.save('states', states.transpose())
