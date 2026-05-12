"""
Test TUFTTE parameter learning with partial data (6 train files + 2 test files).
This allows for faster testing before running on full dataset.
"""

import torch
from utils.NetworkParser import parse_topology, parse_histories, parse_tunnels
from utils.scenario import subScenarios
from algorithms.TUFTTESolver_old import TUFTTESolver, Availability

def test_with_partial_data():
    print("="*70)
    print("Testing Parameter Learning TUFTTE with Partial Data")
    print("="*70)
    
    # Setup network
    print("\n1. Loading network topology...")
    network = parse_topology("Abilene")
    
    print("\n2. Loading partial history data...")
    # Use 6 training files (6 weeks ≈ 1.5 months) and 2 test files (2 weeks)
    parse_histories(network, num_train_files=6, num_test_files=2)
    
    print("\n3. Generating tunnels...")
    parse_tunnels(network, k=8)
    network.tunnel_type = "int"
    network.scale = 1.0
    network.prepare_solution_format()
    
    # Setup scenarios
    print("\n4. Setting up failure scenarios...")
    prob_failure = []
    edge_included = []
    for edge in network.edges:
        if set(edge) in edge_included:
            continue
        prob_failure.append(network.edges[edge].prob_failure)
        edge_included.append(set(edge))
    
    scenarios_all = subScenarios(prob_failure, cutoff=1e-6)
    network.set_scenario(scenarios_all)
    
    print(f"\nNetwork Summary:")
    print(f"  Name: {network.name}")
    print(f"  Nodes: {len(network.nodes)}")
    print(f"  Edges: {len(network.edges)}")
    print(f"  Tunnels: {len(network.tunnels)}")
    print(f"  Scenarios: {len(network.scenarios)}")
    print(f"  Train TMs: {len(network.train_hists._tms)} (from 6 .hist files)")
    print(f"  Test TMs: {len(network.test_hists._tms)} (from 2 .hist files)")
    
    # Initialize solver
    print("\n5. Initializing TUFTTE solver...")
    solver = TUFTTESolver(network, hist_len=12, type=Availability, suffix="_quick")
    
    print("\n6. Starting training and testing...")
    print("   This will:")
    print("   - Pre-train parameter network (theta → 1.0)")
    print("   - Train full model (DemandLossModel or TEAVARModel)")
    print("   - Run on test data")
    print("\n   Note: Training on 6 files (~12,096 TMs) will take some time...")
    
    solver.solve()
    
    print(f"\n7. Results:")
    print(f"   Solutions generated: {len(network.solutions.solutions)}")
    print(f"   Model saved to: data/{network.name}/model_parameter_quick.pkl")
    print(f"                   data/{network.name}/model_A_quick.pkl (or model_D_quick.pkl)")
    print(f"   Optimal values cached in: data/{network.name}/opts_A_quick/int_1.0.opt")
    
    print("\n✓ Partial data test completed successfully!")
    print("\nNext steps:")
    print("  - Check availability metrics in network.solutions")
    print("  - Review theta parameter values")
    print("  - If results look good, run on full 18+6 files dataset")

if __name__ == "__main__":
    try:
        test_with_partial_data()
    except Exception as e:
        print(f"\n✗ Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
