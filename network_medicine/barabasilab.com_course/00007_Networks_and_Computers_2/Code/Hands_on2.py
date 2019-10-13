# conda activate py36_network_medicine && \
# cd /mnt/external_disk/mycodehtml/bio_health/Network_medicine/barabasilab.com_course/Networks_and_Computers_2 && \
# rm e.l && python Hands_on2.py \
# 2>&1 | tee -a e.l && code e.l

# ================================================================================
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'Networks_and_Computers_2'))
	print(os.getcwd())
except:
	pass

from IPython import get_ipython

# ================================================================================
# Network generation, paths, connectivity

# - We're going to write our own Erdos-Renyi generation algorithm 
# and use NetworkX to demonstrate some of the concepts and algorithms in the textbook

# ================================================================================
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
# get_ipython().run_line_magic('matplotlib', 'inline')

# a deque is like a list, but it supports O(1) time insertion/removal at either end
from collections import deque
import itertools as it

# ================================================================================
# change default setting to be less ugly
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
# Generate an ER (erdos renyi) network

# - "ER model G(n,p)" is parameterized by "number of nodes n" and "connection probability p"
# - We will implement "ER model G(n,p)" as a function that takes n, p as arguments

def erdos_renyi(n, p):
    # Create an empty graph
    G = nx.Graph()
 
    # add "n number of nodes"
    G.add_nodes_from(range(0, n))
 
    # for all possible pairs of nodes, add a link with probability p
    for node1 in range(0, n):
        for node2 in xrange(node1 + 1, n):
            if np.random.uniform() < p:
                G.add_edge(node1, node2)
    return G

# ================================================================================
# "ER model G(n,p)" by using more pythonic way

def erdos_renyi(n, p):
    G = nx.Graph()
    nodes = range(n)
    G.add_nodes_from(nodes)
    G.add_edges_from(edge for edge in it.combinations(nodes, 2) if np.random.uniform() < p)
    return G

# ================================================================================
G = erdos_renyi(n=10**3, p=1e-2)
# print("Number of nodes in the graph",len(G))
# print("Number of edges in the graph",G.size())
# Number of nodes in the graph 1000
# Number of edges in the graph 5030

degrees = [G.degree(node) for node in G]
# print("Average number of degree in the graph G",np.mean(degrees))
# Average number of degree in the graph G 10.06

# ================================================================================
# Do "these statistics" make sense for the "n" and "p" which we provided?

# ================================================================================
# Breadth first search

def bfs(G, source):
    """ return a dictionary that maps "node-->distance" for all nodes reachable from the source node, 
    in the unweighted undirected graph G """
 
    # nodes which are left to visit
    nodes = deque()
    nodes.append(source)
 
    # ================================================================================
    # dictionary that gives True or False for each node

    visited={}
    for node in G:
      visited[node]=False
 
    visited[source] = True
 
    # ================================================================================
    # "Initial distances to source" are: 
    # "0" for source itself
    # "infinity" otherwise

    dist={}
    for node in G:
      dist[node]=np.inf

    dist[source] = 0

    # ================================================================================
    while nodes:
     
        # take the "earliest-added element" to the deque (why do we do this instead of popright?)
        node = nodes.popleft()
     
        # visit all neighbors unless they've been visited, record their distances
        for nbr in G.neighbors(node):
            if not visited[nbr]: # if that node is not visited yet
                dist[nbr] = dist[node] + 1
                visited[nbr] = True # change the status as "visited"
                nodes.append(nbr)
    return dist

# ================================================================================
# Components

# As explained in the slides, we can use "BFS mutiple times" to get "all the components"

def components(G):
    """
    return a list of tuples, 
    where each tuple is the nodes in a component of G
    """
    components = []
 
    nodes_left = set(G.nodes())
    while nodes_left:
        src = nodes_left.pop()
        dist = bfs(G, src)
        component = tuple(node for node in dist.keys() if dist[node] < np.inf)
        components.append(component)
        nodes_left = nodes_left - set(component)
    return components

# ================================================================================
# Let's test it on the "100 node network" we generated above

# get list of all components
C = components(G)
# print("C",C)

# sort "the components" by size in descending order
C = sorted(C, key=lambda c: len(c), reverse=True)
# print("C",C)

# print the lengths of the components
component_sizes = [len(c) for c in C]
print(component_sizes)
# [1000]

# ================================================================================
# Hands-on

# * Let's use what we've learned today to reproduce some of the results on pg. 16 of Chapter 3: Random Networks

# * I need 4 groups: Subcritical, Critical, Supercritical, and Connected

# * You can use the networkx function "fast_gnp_random_graph(n, p)" to generate your networks

# ================================================================================
# Subcritical, Critical, and Supercritical groups

# You will consider networks of "average degree" $k = 0.5, k=1, k=2$ respectively. Choose "connection probabilities" accordingly. 

# Each group must do the following:

# * Generate one random network each of sizes $N=10^2, 10^3, 10^4, 10^5, 10^6$ using your group's average degree

# * For each network, use the code above to get the connected components. 

# * Following the lesson on plotting distributions in the last lecture, modify the code from the last lecture to
# plot the distribution of the sizes of the connected components in log-log scale. 
# Plot this for all the networks in the same figure using different colors. 

# * Calculate the size of the largest component for each of the 5 networks. 
# Are they giant components? 
# Write new code to plot the largest component size as a function of N in semi-log scale (hint: use "plt.semilogx")

# * Compare the above two results with your expectation from the book

# ================================================================================
# Connected group

# You will consider networks of "average degree" $k = 20$; choose "connection probabilities" accordingly. 
# You must do the following:

# * Generate one random network each of sizes $N=10^2, 10^3, 10^4, 10^5, 10^6$ with this average degree (k = 20).

# * Use "%timeit" on BFS for the network of size $10^6$ (you can pick an arbitrary source node. Why is that?) 
# to get the number of seconds it takes on your laptop to get single-source shortest paths. 
# Using this, give an estimate of how long it would take to calculate the diameter of this network, 
# which would require calculating shortest path lengths for ALL possible pairs.

# * As an alternative, figure out what the following code is doing to calculate the diameter approximately. Explain to the class.

# * Use this code to plot the "pseudo-diameter" as a function of N in semi-log scale (hint: use "plt.semilogx")

# ================================================================================
def pseudo_diameter(G):
    
    # ================================================================================
    # diameter should be "infinity" if not connected
    if not nx.is_connected(G):
        return np.inf
  
    # ================================================================================
    nodes = list(G.nodes())

    # randomly pick a node from G
    u = random.choice(nodes)

    # ================================================================================
    diam = 0
    while True:
        # what do you think this does? You are allowed to read the NetworkX documentation
        d = nx.single_source_shortest_path_length(G, u)
     
        # get farthest node from u & the corresponding distance
        farthest_node, d_max = max(d.items(), key=lambda item: item[1])
        if d_max <= diam:
            return diam
        else:
            u = farthest_node
            diam = d_max

# ================================================================================
# Don't reinvent the wheel

# Now that you can understand how some basic graph analysis algorithms work, 
# you should never use them again and instead you will use the following commands which are better written and faster.

# ================================================================================
# Graph generation

# erdos_renyi_graph(n, p)

# ================================================================================
# Generate a random graph. More or less the same as we implemented above.
# fast_gnp_random_graph(n, p)

# ================================================================================
# Algorithms which are much faster for sparse graphs

# Algorithms for managing "paths" and "path length"

# All of the below functions work on both "Graph" and "DiGraph" objects

# ================================================================================
# test to see if there is a path (of any length) in G from "source" to "dest"
# has_path(G, source, dest)

# returns path as a sequence of nodes
# shortest_path(G, source, dest)

# only returns the length
# shortest_path_length(G, source, dest)

# same as above, but gives ALL shortest paths
# all_shortest_paths(G, source, dest)

# return dictionary "d" where "d[node]" is respectively, 
# the shortest path/path length from "source" to "node"
# single_source_shortest_path(G, source)
# single_source_shortest_path_length(G, source)

# return dictionary "d" where "d[node1][node2]" is as above
# all_pairs_shortest_path(G)
# all_pairs_shortest_path_length(G)

# As above, but for weighted "Graph/DiGraph" objects
# dijkstra_path(G, source, dest)
# dijkstra_path_length(G, source, dest)

# As above, but for weighted "Graph/DiGraph" objects  
# single_source_dijkstra_path(G, source)
# single_source_dijkstra_path_length(G, source)

# ================================================================================
# Searching

# All of the below work on both "Graph" and "DiGraph" objects

# ================================================================================
# Return a Di/Graph representing the tree spanned by a breadth-first search starting at "source"
# bfs_tree(G, source)

# Same using depth-first search (gives same result)
# dfs_tree(G, source)

# same as above, but gives ALL shortest paths
# all_shortest_paths(G, source, dest)

# ================================================================================
# Connectedness (Undirected)

# The below work only on "Graph" objects

# ================================================================================
# "True" or "False" depending on whether "G" is connected or not   
# is_connected(G)

# Return a list of lists, where each sub-list contains the nodes in one component
# connected_components(G)  

# number_connected_components(G)
# Returns only the length of the list above
# connected_component_sugraphs(G)
# Returns a list of new "Graph" objects each representing a component of "G"
# node_connected_component(G, node)
# Return a list of the nodes in the component of "G" containing "node"

# ================================================================================
# Connectedness (Strong and weak)

# The commands below work only on "DiGraph" objects

# Note: the "is_weakly_" versions are equivalent to first converting the DiGraph to undirected using G.undirected(), and then running the undirected equivalents above.

# is_strongly_connected(G)
# strongly_connected_components(G)  
# number_strongly_connected_components(G)
# strongly_connected_component_sugraphs(G)  

# is_weakly_connected(G)
# weakly_connected_components(G)  
# number_weakly_connected_components(G)
# weakly_connected_component_sugraphs(G)

# All are analogous to the undirected case

# ================================================================================
# How to generate networks with "scale-free degree distributions"

# * Let's use the "configuration model" to generate networks with (approximately) scale-free degree distributions.
# * configuration_model in NetworkX produces a MultiGraph. Why? We will change that into a regular graph.

# Remember, the configuration model takes as input a "desired degree sequence". 
# It then spits out a network having that degree sequence. 
# So first, we need to be able to randomly generate degrees following a power-law distribution.

def powerlaw_degree_sequence(n, gamma, k_min):
    """
    Generates power-law distributed numbers from uniformly-distributed numbers 
    described in Clauset et al., 2009, appendix D
    """

    r = np.random.uniform(0, 1, size=n)
    deg = np.floor((k_min-0.5)*(1.0 - r)**(-1.0/(gamma-1)) + 0.5)
    deg = list(map(int, deg))
    return deg

# ret=powerlaw_degree_sequence(n=100, gamma=0.3, k_min=2)
# print("ret",ret)
# [0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1]

# ================================================================================
def sf_net(n, gamma, k_min):
    deg = powerlaw_degree_sequence(n, gamma, k_min)

    # ================================================================================
    # "sum of all degrees" must be even. Why is that?
    if sum(deg) % 2 == 1: # if sum of "deg" is odd
        deg[0] += 1 # make it "even"

    # ================================================================================
    # configure the model with "power-law distributed deg"
    G = nx.configuration_model(deg)

    # ================================================================================
    H = nx.Graph(G)

    # ================================================================================
    H.remove_edges_from(H.selfloop_edges())

    # ================================================================================
    return H

# ================================================================================
# Hands-on exercise
# - Split into 4 groups, one each for gamma = 2.001, 2.5, 3, and 3.5
# - All groups should use the above code to generate networks of sizes $10^2 \ldots 10^5$ as before with their chosen scaling exponent. 

# Generate all networks with "minimum degree cutoff" $k_{min} = 5$

# * First, measure the "maximum degree" of each network $k_{max}$, and then plot kmax in log-log scale as a function of $N$ 
# * Next, I want you to plot the average shortest-path distance as a function of $N$ in semi-log scale. 
# * Note that for larger networks it will be impossible to measure all pairs shortest paths. 
# As an approximation, you should take a random *sample* of pairs of nodes (src, dest), 
# measure the shortest path length between src and dest, and then take the average. Use 100 random node pairs per network.

# * Hint 1: [np.random.choice(G, size=2, replace=False) for _ in range(100)] will give you a list of 100 random node pairs from G

# * Hint 2: You will need to run this within one component. Choose the largest. 

# The following code will sort the components from largest to smallest
# components = sorted(components, key=len, reverse=True)

# You can then use the subgraph command on the first component (components[0])

# * Hint 3: Use "nx.shortest_path_length" and "nx.connected_components". 
# They are faster than what we've written.
