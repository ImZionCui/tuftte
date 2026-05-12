"""
Quick test: Run TUFTTE with parameter learning and check basic functionality.
"""

import torch
from utils.NetworkParser import parse_topology, parse_histories, parse_tunnels
from utils.scenario import subScenarios
from algorithms.TUFTTESolver_old import TUFTTESolver, Availability

def quick_test():
    print("="*60)
    print("Quick Test: Parameter Learning TUFTTE")
    print("="*60)
    
    # Minimal setup
    network = parse_topology("Abilene")
    parse_histories(network, num_train_files=6, num_test_files=2)  # Use 6 train files and 2 test files
    network.reduce_data(1000, 30)  # Use minimal data for quick test
    parse_tunnels(network, k=8)
    network.tunnel_type = "int"
    network.scale = 1.0
    network.prepare_solution_format()
    
    # Setup scenarios
    prob_failure = []
    edge_included = []
    for edge in network.edges:
        if set(edge) in edge_included:
            continue
        prob_failure.append(network.edges[edge].prob_failure)
        edge_included.append(set(edge))
    
    scenarios_all = subScenarios(prob_failure, cutoff=1e-6)
    network.set_scenario(scenarios_all)
    
    print(f"\nNetwork: {network.name}")
    print(f"Train TMs: {len(network.train_hists._tms)}")
    print(f"Test TMs: {len(network.test_hists._tms)}")
    
    # Test
    print("\nInitializing solver...")
    solver = TUFTTESolver(network, hist_len=12, type=Availability, suffix="_quick")
    
    print("Running solve()...")
    solver.solve()
    
    print(f"\nSolutions generated: {len(network.solutions.val)}")
    print("\n✓ Test passed!")

if __name__ == "__main__":
    quick_test()
