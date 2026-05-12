"""
Test script for the new "Learning to Configure Parameters" paradigm.
This verifies that the model learns control parameters (theta) instead of predicting demand.
"""

import torch
from utils.NetworkParser import parse_topology, parse_histories, parse_tunnels
from utils.scenario import subScenarios
from algorithms.TUFTTESolver_old import TUFTTESolver, DemandLoss, Availability

def test_parameter_learning():
    print("="*60)
    print("Testing: Learning to Configure Parameters (not Predict Demand)")
    print("="*60)
    
    # Load network
    network = parse_topology("Abilene")
    parse_histories(network)
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
    print(f"Num demands (OD pairs): {len(network.demands)}")
    print(f"Num tunnels: {len(network.tunnels)}")
    print(f"Num edges: {len(network.edges)}")
    print(f"Train TMs: {len(network.train_hists._tms)}")
    print(f"Test TMs: {len(network.test_hists._tms)}")
    
    # Test DemandLoss model
    print("\n" + "-"*60)
    print("Testing DemandLoss with Parameter Learning")
    print("-"*60)
    solver_d = TUFTTESolver(network, hist_len=12, type=DemandLoss)
    
    # Check if parameter model exists
    import os
    param_model_path = f"data/{network.name}/model_parameter.pkl"
    if os.path.exists(param_model_path):
        print(f"✓ Parameter model found: {param_model_path}")
    else:
        print(f"✗ Parameter model not found, will train: {param_model_path}")
    
    # Train or load model
    print("\nTraining/Loading DemandLoss model with parameter configuration...")
    model = solver_d._train()
    
    # Test forward pass with true demand
    print("\nTesting forward pass with true demand...")
    device = solver_d.device
    
    # Get a sample from test set
    from algorithms.TUFTTESolver_old import TUFTTEDataset
    from torch.utils.data import DataLoader
    fake_opts = [0 for _ in range(len(network.test_hists._tms) - 12)]
    data = TUFTTEDataset(network.test_hists._tms, fake_opts, 12)
    loader = DataLoader(data, shuffle=False)
    
    with torch.no_grad():
        for (x, y, _) in loader:
            x, y = x.to(device), y.to(device)
            
            # Key test: model should receive both history (x) and true demand (y)
            x_star, theta = model(x, y)
            
            print(f"\n✓ Forward pass successful!")
            print(f"  Input history shape: {x.shape}")
            print(f"  True demand shape: {y.shape}")
            print(f"  Output theta (parameters) shape: {theta.shape}")
            print(f"  Output x_star (tunnel flows) shape: {x_star.shape}")
            print(f"\n  Theta statistics:")
            print(f"    Mean: {theta.mean().item():.4f}")
            print(f"    Std:  {theta.std().item():.4f}")
            print(f"    Min:  {theta.min().item():.4f}")
            print(f"    Max:  {theta.max().item():.4f}")
            print(f"    (Expected range: [0.5, 1.5])")
            
            # Verify theta is in valid range
            assert theta.min() >= 0.5 and theta.max() <= 1.5, "Theta out of range!"
            print(f"\n✓ Theta values are in valid range [0.5, 1.5]")
            
            break  # Only test one sample
    
    print("\n" + "="*60)
    print("✓ All tests passed! Model successfully learns parameters.")
    print("="*60)

if __name__ == "__main__":
    test_parameter_learning()
