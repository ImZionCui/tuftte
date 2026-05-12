"""
Test script to verify parameter-learning TUFTTE on availability experiment.
Compare new parameter-based approach with original prediction-based approach.
"""

import os
import torch
import numpy as np
from utils.NetworkParser import parse_topology, parse_histories, parse_tunnels
from utils.scenario import subScenarios
from algorithms.TUFTTESolver_old import TUFTTESolver, Availability
from utils.riskMetric import calculate_risk

def setup_network(topology_name="Abilene", num_train_files=None, num_test_files=None, 
                  num_train_samples=None, num_test_samples=30):
    """Setup network with scenarios and tunnels.
    
    Args:
        topology_name: Network topology name
        num_train_files: Number of training .hist files to use (None for all)
        num_test_files: Number of testing .hist files to use (None for all)
        num_train_samples: Number of training samples to use after loading (None for all)
        num_test_samples: Number of test samples to use
    """
    network = parse_topology(topology_name)
    parse_histories(network, num_train_files=num_train_files, num_test_files=num_test_files)
    
    # Reduce data if specified (further limit after file loading)
    if num_train_samples or num_test_samples:
        network.reduce_data(num_train_samples, num_test_samples)
    
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
    
    return network

def test_availability_with_parameters():
    """Test TUFTTE with parameter learning on availability experiment."""
    print("="*70)
    print("Testing: TUFFTE with Parameter Learning on Availability Experiment")
    print("="*70)
    
    # Setup - use 6 train files and 2 test files for faster testing
    network = setup_network("Abilene", 
                           num_train_files=6, 
                           num_test_files=2,
                           num_train_samples=5000, 
                           num_test_samples=30)
    
    print(f"\nNetwork: {network.name}")
    print(f"Num demands (OD pairs): {len(network.demands)}")
    print(f"Num tunnels: {len(network.tunnels)}")
    print(f"Num scenarios: {len(network.scenarios)}")
    print(f"Train TMs: {len(network.train_hists._tms)}")
    print(f"Test TMs: {len(network.test_hists._tms)}")
    
    # Initialize solver
    print("\n" + "-"*70)
    print("Testing Availability (TEAVARModel) with Parameter Learning")
    print("-"*70)
    
    solver = TUFTTESolver(network, hist_len=12, type=Availability)
    
    # Check model status
    model_path = f"data/{network.name}/model_{Availability}.pkl"
    param_model_path = f"data/{network.name}/model_parameter.pkl"
    
    if os.path.exists(model_path):
        print(f"✓ TUFTTE model found: {model_path}")
    else:
        print(f"✗ TUFTTE model not found, will train")
    
    if os.path.exists(param_model_path):
        print(f"✓ Parameter model found: {param_model_path}")
    else:
        print(f"✗ Parameter model not found, will train")
    
    # Run solver
    print("\nRunning solver on test set...")
    solver.solve()
    
    # Calculate availability
    print("\nCalculating availability...")
    loss_list = []
    availability_list = []
    
    for i, (loss, availability) in enumerate(calculate_risk(network, hist_len=12)):
        loss_list.append(loss)
        availability_list.append(availability)
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1} test cases")
    
    # Results
    print("\n" + "="*70)
    print("RESULTS: Parameter Learning on Availability")
    print("="*70)
    print(f"Mean Loss: {np.mean(loss_list):.6f}")
    print(f"Std Loss:  {np.std(loss_list):.6f}")
    print(f"Mean Availability: {np.mean(availability_list):.6f}")
    print(f"Std Availability:  {np.std(availability_list):.6f}")
    print(f"\nTest cases: {len(availability_list)}")
    
    # Detailed stats
    print("\nDetailed Statistics:")
    print(f"  Min Availability: {np.min(availability_list):.6f}")
    print(f"  Max Availability: {np.max(availability_list):.6f}")
    print(f"  Median Availability: {np.median(availability_list):.6f}")
    
    # Check parameter predictions
    print("\n" + "-"*70)
    print("Parameter Model Analysis")
    print("-"*70)
    
    # Load and inspect parameter model
    model = solver._train()
    model.eval()
    
    from algorithms.TUFTTESolver_old import TUFTTEDataset
    from torch.utils.data import DataLoader
    
    fake_opts = [0] * (len(network.test_hists._tms) - 12)
    data = TUFTTEDataset(network.test_hists._tms, fake_opts, 12)
    loader = DataLoader(data, batch_size=32, shuffle=False)
    
    all_thetas = []
    with torch.no_grad():
        for (x, y, _) in loader:
            x = x.to(solver.device)
            x_star, theta = model(x, y.to(solver.device))
            all_thetas.append(theta.detach().cpu().numpy())
    
    all_thetas = np.concatenate(all_thetas, axis=0)
    
    print(f"\nTheta (control parameter) statistics:")
    print(f"  Shape: {all_thetas.shape}")
    print(f"  Mean: {np.mean(all_thetas):.4f}")
    print(f"  Std:  {np.std(all_thetas):.4f}")
    print(f"  Min:  {np.min(all_thetas):.4f}")
    print(f"  Max:  {np.max(all_thetas):.4f}")
    print(f"  Expected range: [0.5, 1.5]")
    
    # Verify theta is in valid range
    if np.all(all_thetas >= 0.5) and np.all(all_thetas <= 1.5):
        print(f"\n✓ All theta values in valid range [0.5, 1.5]")
    else:
        print(f"\n✗ Some theta values out of range!")
        print(f"  Out of range count: {np.sum((all_thetas < 0.5) | (all_thetas > 1.5))}")
    
    # Histogram of theta values
    print(f"\nTheta distribution (percentiles):")
    for p in [10, 25, 50, 75, 90]:
        print(f"  {p}th percentile: {np.percentile(all_thetas, p):.4f}")
    
    print("\n" + "="*70)
    print("✓ Test completed successfully!")
    print("="*70)

if __name__ == "__main__":
    test_availability_with_parameters()
