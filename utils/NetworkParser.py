"""
Parse topology, history, tunnel, etc, from files.
"""

import random
import csv
import re
from glob import glob
from numpy.random import weibull

from .NetworkTopology import Network, Histories, BPS_TO_MBPS

def parse_topology(network_name, use_weibull=False, shape=0.8, weibull_scale=0.001):
    """
    Parse topology.

    parameters:
        network_name(str): the name of topology;
        use_weibull(bool): whether to use the weibull distribution to generate probabilities;
        shape(float): shape parameter of weibull distribution;
        weibull_scale(float): scale parameter of weibull distribution.

    return:
        network(Network): the Network structure of topology.
    """
    network = Network(network_name)
    with open(f"data/{network_name}/{network_name}_int.pickle.nnet") as fi:
        reader = csv.reader(fi)
        for row in reader:
            to_node = row[0]
            from_node = row[1]
            capacity = float(row[2]) * BPS_TO_MBPS
            # generate the probability of link failure
            if len(row) < 4 or use_weibull:
                prob_failure = weibull(shape) * weibull_scale
            else:
                prob_failure = float(row[3])
            network.add_node(to_node, None, None)
            network.add_node(from_node, None, None)
            network.add_edge(from_node, to_node, 200, capacity, prob_failure)
            network.add_edge(to_node, from_node, 200, capacity, prob_failure)
            
    return network

def parse_demands(network, scale=0.25):
    num_nodes = len(network.nodes)
    demand_matrix = {}
    with open(f"data/{network.name}/demand.txt") as fi:
        reader = csv.reader(fi, delimiter=' ')
        for row_ in reader:
            if row_[0] == 'to_node': continue
            row = [float(x) for x in row_ if x]
            assert len(row) == num_nodes ** 2
            for idx, dem in enumerate(row):
                from_node = int(idx/num_nodes) + 1
                to_node = idx % num_nodes + 1
                assert str(from_node) in network.nodes
                assert str(to_node) in network.nodes
                if from_node not in demand_matrix:
                    demand_matrix[from_node] = {}
                if to_node not in demand_matrix[from_node]:
                    demand_matrix[from_node][to_node] = []
                demand_matrix[from_node][to_node].append(dem/1000.0)
        for from_node in demand_matrix:
            for to_node in demand_matrix[from_node]:
                max_demand = demand_matrix[from_node][to_node][0]
                network.add_demand(str(from_node), str(to_node), max_demand, scale)
    if network.tunnels:
        for t in network.tunnels.values():
            tunnel_start = t.path[0].e[0]
            tunnel_end = t.path[-1].e[-1]
            if (tunnel_start, tunnel_end) in network.demands:
                demand = network.demands[(tunnel_start, tunnel_end)]
                demand.add_tunnel(t)

        remove_demands_without_tunnels(network)

def parse_histories(network, scale=1, num_train_files=None, num_test_files=None):
    """
    Parse histories into Network structure.

    parameters:
        network(Network): the Network without histories;
        scale(float): the scale factor of demands.
        num_train_files(int): number of training .hist files to use (None for all)
        num_test_files(int): number of testing .hist files to use (None for all)
    """
    num_nodes = len(network.nodes)
    train_hists = sorted(glob(f"data/{network.name}/train" + "/*.hist"))
    test_hists = sorted(glob(f"data/{network.name}/test" + "/*.hist"))
    
    # Limit number of files if specified
    if num_train_files is not None:
        train_hists = train_hists[:num_train_files]
        print(f"Using {len(train_hists)} training .hist files: {[h.split('\\')[-1] for h in train_hists]}")
    if num_test_files is not None:
        test_hists = test_hists[:num_test_files]
        print(f"Using {len(test_hists)} testing .hist files: {[h.split('\\')[-1] for h in test_hists]}")
    
    network.train_hists = Histories(train_hists, "train", num_nodes)
    network.test_hists = Histories(test_hists, "test", num_nodes)
    network.set_scale(scale)
    
def parse_tunnels(network, paths="ksp", k=5):
    """
    Parse tunnels.

    parameters:
        network(Network): Network structure;
        paths(str): the way to collect paths;
        k(int): if k-shortest-path (KSP) is used, k is the number of distinct shortest paths.
    """
    if paths == "ksp":
        network.tunnel_type = "ksp_" + str(k)
        for node1 in network.nodes:
            for node2 in network.nodes:
                if node1 == node2: continue
                paths = network.k_shortest_paths(node1, node2, k)
                for path in paths:
                    network.add_tunnel(path)
    else:
        network.tunnel_type = paths
        parse_paths(network, paths)
    
    if network.demands:
        remove_demands_without_tunnels(network)

def remove_demands_without_tunnels(network):
    """
    If there is not a tunnel between a pair of nodes, we delete the traffic demand between them.
    """
    removable_demands = [p for p, d in network.demands.items() if not d.tunnels or d.amount == 0]
    for demand_pair in removable_demands:
        del network.demands[demand_pair]

def initialize_weights(network):
    for tunnel in network.tunnels.values():
        tunnel.add_weight(random.randint(1, 10))

def parse_paths(network, paths):
    """
    Parse paths from file.
    """
    print("Loading paths (%s)......" % paths)
    with open(f"data/{network.name}/paths/{paths}") as fi:
        dst = ""
        regex = re.compile(r'h(\d+) -> h(\d+)')
        for row in fi:
            if "->" in row:
                mo = regex.search(row)
                dst = mo.group(2)
            elif row != '\n':
                t = []
                numbers = re.findall(r'\d+', row.split(']')[0])
                l = len(numbers)
                i = 0
                while i < l:
                    if numbers[i] == numbers[i + 1]:
                        i += 2
                        continue
                    t.append(numbers[i])
                    i += 2
                t.append(dst)
                network.add_tunnel(t)