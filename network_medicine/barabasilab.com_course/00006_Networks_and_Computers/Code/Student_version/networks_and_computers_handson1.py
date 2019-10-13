# conda activate py36_network_medicine && \
# cd /mnt/external_disk/mycodehtml/bio_health/Network_medicine/barabasilab.com_course/Networks_and_Computers/Code/Student_version && \
# rm e.l && python networks_and_computers_handson1.py \
# 2>&1 | tee -a e.l && code e.l

# ================================================================================
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'Networks_and_Computers/Code/Student_version'))
	print(os.getcwd())
except:
	pass

from IPython import get_ipython

# ================================================================================
# Importing required modules
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
# get_ipython().run_line_magic('matplotlib', 'inline')

# ================================================================================
# change defaults to be less ugly
mpl.rc('xtick', labelsize=14, color="#222222") 
mpl.rc('ytick', labelsize=14, color="#222222") 
mpl.rc('font', **{'family':'sans-serif','sans-serif':['Arial']})
mpl.rc('font', size=16)
mpl.rc('xtick.major', size=6, width=1)
mpl.rc('xtick.minor', size=3, width=1)
mpl.rc('ytick.major', size=6, width=1)
mpl.rc('ytick.minor', size=3, width=1)
mpl.rc('axes', linewidth=1, edgecolor="#222222", labelcolor="#222222")
mpl.rc('text', usetex=False, color="#222222")

# ================================================================================
# NetworkX provides the following "classes" (that represent network-related data)
# as well as "network analysis algorithms" (that operate on these objects):

# Graph        - Undirected graph with self loops
# DiGraph      - Directed graph with self loops
# MultiGraph   - Undirected Graph with self loops and multiple edges
# MultiDiGraph - Directed Graph with self loops and multiple edges

# ================================================================================
print(nx.__version__)
# 2.3

# ================================================================================
# Create an "empty", "undirected" network

G = nx.Graph()

# ================================================================================
# Nodes can be almost anything (numbers, strings, GPS coordinates, etc)
# Nodes can be added one at a time:

# Add "0 node" to "graph G"
G.add_node(0)

# Add "John node" to "graph G"
G.add_node("John")

# tuple object representing, say, longitude and latitude
pos = (1.2, 3.4) 
G.add_node(pos)

# ...or many at once from a python container
# [1,2,3] is a list containing 1, 2, 3
G.add_nodes_from([1, 2, 3])

# ================================================================================
# Nodes can have arbitrary attributes (which are associated with nodes), (which are contained in a string-index dictionary)

# Add "Louis node" + "Attributes of node"
G.add_node("Louis", eye_color='blue', height=6)

# Add "Laszlo node"
G.add_node("Laszlo")
# Add "attributes" to an "existing node"
G.nodes["Laszlo"]["citations"] = 10**6

# ================================================================================
# Get values from nodes

Louis_node_eye_color_attribute=G.nodes["Louis"]["eye_color"]
Louis_node_height_attribute=G.nodes['Louis']['height']
Laszlo_node_citations_attribute=G.nodes["Laszlo"]["citations"]
# print("Louis_node_eye_color_attribute",Louis_node_eye_color_attribute)
# print("Louis_node_height_attribute",Louis_node_height_attribute)
# print("Laszlo_node_citations_attribute",Laszlo_node_citations_attribute)
# blue
# 6
# 1000000

# ================================================================================
# An edge between node1 and node2: (node1, node2)  

# Edges can be added one at a time

# Add "edge" between "node 0" and "node 1"
G.add_edge(0, 1)

# ================================================================================
# Add "multiple edges" 

edge_list = [ (2, 1), ("Louis", "Laszlo"), (3, 4) ]
G.add_edges_from(edge_list)

# ================================================================================
# "Nodes" will be "automatically created" if nodes don't already exist.

# ================================================================================
# Edge attributes

# Edges can have arbitrary attributes. 

# ================================================================================
# Add "edge ('Louis', 'Sebastian')" + "edge attribute weight=10"
G.add_edge("Louis", "Sebastian", weight=10)

G.add_edge("Hopkinton", "Boston")
G.edges["Hopkinton", "Boston"]['distance'] = 26.2

# ================================================================================
# Basic operations

# ================================================================================
# Size of the network

# number of nodes of "graph G"
# print("G.number_of_nodes()",G.number_of_nodes())
# 12

# number of nodes of "graph G"
# print("len(G)",len(G))
# 12

# number of edges of "graph G"
# print("G.number_of_edges()",G.number_of_edges())
# 6

# number of edges of "graph G"
# print("G.size()",G.size())
# 6

# ================================================================================
# How to see whether "nodes" exist
res=G.has_node("Louis")
# print("res",res)
# True

# How to see whether "nodes" exist
res="Sebastian" in G
# print("res",res)
# True

# ================================================================================
# How to see whether "edges" exist

ret=G.has_edge(3, 4)
# print("ret",ret)
# True

ret=G.has_edge("Louis", 0)
# print("ret",ret)
# False

# ================================================================================
# How to find neighbaors of a node

neighbors_of_node_1_in_graph_G=list(G.neighbors(1))
# print("neighbors_of_node_1_in_graph_G",neighbors_of_node_1_in_graph_G)
# [0, 2]

# ================================================================================
# * In `DiGraph` objects, `G.neighbors(node)` gives the successors of `node`, as does `G.successors(node)`  
# * Predecessors of `node` can be obtained with `G.predecessors(node)`

# ================================================================================
# How to iterate over nodes using G.nodes()

# for node, data in G.nodes(data=True): # data=True includes "node attributes" as dictionaries
#     print(node)
#     print(data)
#     print("")

# ================================================================================
# How to iterate over edges using G.edges()

# for n1, n2, data in G.edges(data=True):
#     print([n1,n2])
#     print(data)
#     print("")

# ================================================================================
# Calculate degrees

degree_of_node_Louis=G.degree("Louis")
# print("degree_of_node_Louis",degree_of_node_Louis)

degree_of_all_nodes_in_graph_G=G.degree()
# print("degree_of_all_nodes_in_graph_G",degree_of_all_nodes_in_graph_G)
# [(0, 1), ('John', 0), ((1.2, 3.4), 0), (1, 2), (2, 1), (3, 1), ('Louis', 2), ('Laszlo', 1), (4, 1), ('Sebastian', 1), ('Hopkinton', 1), ('Boston', 1)]

degree_of_all_nodes_list=[]
for one_node in G:
  degree_of_one_node=G.degree(one_node)
  degree_of_all_nodes_list.append(degree_of_one_node)
# print("degree_of_all_nodes_list",degree_of_all_nodes_list)
# [1, 0, 0, 2, 1, 1, 2, 1, 1, 1, 1, 1]

# ================================================================================
# In directed graphs (DiGraph), there are two types of degree.

# in_degree_of_node=directed_graph_G.in_degree(node)
# out_degree_of_node=directed_graph_G.out_degree(node)
# out_degree_of_node=directed_graph_G.degree()

# ================================================================================
# Other operations

# ================================================================================
# subgraph of graph G, (which are induced by nodes in nbunch)
# subgraph(G,nbunch)
# G.subgraph(nbunch)

# ================================================================================
# DiGraph (with edges reversed )
# reverse(G)

# ================================================================================
# union of 2 graphs
# union(G1, G2)

# ================================================================================
# same, but treats nodes of G1, G2 as different 
# disjoint_union(G1, G2)

# ================================================================================
# "result graph" with only the "edges in common" between G1, G2
# intersection(G1, G2)

# ================================================================================
# "result graph" with only the "edges G1 that aren't in G2"
# difference(G1, G2)

# ================================================================================
# copy of G
# copy(G) or G.copy()

# ================================================================================
# the complement graph of graph G 
# complement(G) or G.complement()

# ================================================================================
# undirected version of graph G (a Graph or MultiGraph)
# convert_to_undirected(G) or G.to_undirected()

# ================================================================================
# directed version of G (a DiGraph of MultiDiGraph)
# convert_to_directed(G) or G.to_directed()

# ================================================================================
# "adjacency matrix A" of graph G (in sparse matrix format; to get full matrix, use A.toarray())
# adjacency_matrix(G)

# ================================================================================
# @ Graph I/O

# ================================================================================
# NetworkX can understand the following common graph formats:

# edge lists
# adjacency lists
# GML
# GEXF
# Python 'pickle'
# GraphML
# Pajek
# LEDA
# YAML

# ================================================================================
# Getting started: "reading in" an "edge list"

# "Read in" the file with the "options"
#   comments='#': lines starting with `#` are treated as comments and ignored  
#   create_using=nx.Graph(): use a "Graph object" to hold the data (i.e., network is undirected)  
#   delimiter=' ': data are separated by whitespace
#   nodetype=int: nodes should be treated as integers
#   encoding='utf-8': encoding of the text file containing the edge list is utf-8

# "read in" an "edge list" from the file 'test.txt'
G = nx.read_edgelist('./test.txt', comments='#', create_using=nx.Graph(), delimiter=' ', nodetype=int, encoding='utf-8')

# ================================================================================
# Allowed formats in "edge list file"

# - Node pairs with no data  
# 1 2

# - Node pairs with python dictionary  
# 1 2 {weight:7, color:"green"}

# ================================================================================
# Basic analysis

# A large number of basic analyses can be done using
# - NetworkX + numpy
# - builtin python functions like min, max, etc

# ================================================================================
# Number of nodes
N = len(G)
# print("N",N)
# N 443

# Number of edges
L = G.size()
# print("L",L)
# L 540

# ================================================================================
degrees = [G.degree(node) for node in G]
# print("degrees",degrees)
# degrees [5, 2, 5, 5, 4,

# print("Average degree: ", 2*L/N)
# print("Average degree (alternate calculation)", np.mean(degrees))
# 2.4379232505643342
# 2.4379232505643342

# Minimum degree
kmin = min(degrees)
# print("kmin",kmin)
# kmin 1

# Maximum degree
kmax = max(degrees)
# print("kmax",kmax)
# kmax 8

# ================================================================================
# @ Drawing the network

# using the "force-based" or "spring" layout algorithm
# fig = plt.figure(figsize=(8,8))
# plt.title("undirected graph G \n node point size=10 \n draw this by using draw_spring()")
nx.draw_spring(G, node_size=10)
# plt.show()
# /mnt/external_disk/Capture_temp/2019_10_13_09:09:27.png

# ================================================================================
# using the fcircular layout algorithm
# fig = plt.figure(figsize=(8,8))
# plt.title("undirected graph G \n node point size=10 \n draw this by using draw_circular()")
nx.draw_circular(G, node_size=10)
# plt.show()
# /mnt/external_disk/Capture_temp/2019_10_13_09:11:28.png

# ================================================================================
# Plotting the degree distribution

# Let's plot "degree distribution" in "log scale" first

# numpy can be used to get "logarithmically-spaced bins" between the minimum and maximum degree

# Get "10 logarithmically spaced bins" between kmin and kmax
bin_edges = np.logspace(np.log10(kmin), np.log10(kmax), num=10)
# print("bin_edges",bin_edges)
# [1.         1.25992105 1.58740105 2.         2.5198421  3.1748021   4.         5.0396842     6.34960421 8.        ]

# histogram the data into these bins
density, _ = np.histogram(degrees, bins=bin_edges, density=True)

# ================================================================================
fig = plt.figure(figsize=(6,4))

log_be = np.log10(bin_edges)
# print("log_be",log_be)
# [0.         0.10034333 0.20068666 0.30103    0.40137333 0.50171666   0.60205999 0.70240332 0.80274666 0.90308999]

# "x" should be midpoint (IN LOG SPACE) of each bin
x = 10**((log_be[1:] + log_be[:-1])/2)
# print("x",x)
# [1.12246205 1.41421356 1.78179744 2.2449241  2.82842712 3.56359487 4.48984819 5.65685425 7.12718975]

plt.loglog(x, density, marker='o', linestyle='none')
plt.title("probability distribution of degree k in log domain")
plt.xlabel(r"degree $k$", fontsize=16)
plt.xlabel(r"degree $k$", fontsize=16)
plt.ylabel(r"$P(k)$: probability of degree $k$ occuring", fontsize=16)

# remove right and top boundaries because they're ugly
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

# Show the plot
# plt.show()

# ================================================================================
# This is clearly not a network withe anything like a heavy-tailed or power law degree distribution.

# ================================================================================
# Let's also plot it in linear-linear scale.

# The `linspace` command in `numpy` is used to get linearly-spaced numbers between two extremes


# Get 20 logarithmically spaced bins between kmin and kmax
bin_edges = np.linspace(kmin, kmax, num=10)

# histogram the data into these bins
density, _ = np.histogram(degrees, bins=bin_edges, density=True)

# ================================================================================
# plot it

fig = plt.figure(figsize=(6,4))

log_be = np.log10(bin_edges)

# "x" should be midpoint (IN LOG SPACE) of each bin
x = 10**((log_be[1:] + log_be[:-1])/2)

plt.plot(x, density, marker='o', linestyle='none')
plt.title("probability distribution of degree k in linear-linear scale")
plt.xlabel(r"degree $k$", fontsize=16)
plt.ylabel(r"$P(k)$: probability of degree $k$ occuring", fontsize=16)

# remove right and top boundaries because they're ugly
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

plt.show()
# /mnt/external_disk/Capture_temp/2019_10_13_10:07:44.png

# ================================================================================
# Hands-on exercise

# Now split into 6 groups, 3 for `example_1.txt` and 3 for `example_2.txt`. Each group should read in their edge

# list file and do the following:

# * Group 1: Do the basic measurements shown above. What can you suspect about the degree distribution of the network just based on the average and extremes in degree?

# * Group 2: Plot the degree distribution in log-log scale. Also plot it in linear scale. Comment on how this fits with the analysis of Group 1.

# * Group 3: Draw the network using the two layout algorithms shown above. How does the the network's appearance echo the findings of groups 1 and 2?

