"""
Compare TUFTTE (Parameter Learning) with other TE algorithms.
Uses partial data for quick comparison testing.
"""

from utils.NetworkParser import parse_topology, parse_histories, parse_tunnels
from utils.prediction import predict_traffic_matrix, RF
from utils.scenario import subScenarios
from utils.GurobiSolver import GurobiSolver
from utils.riskMetric import calculate_risk, validate_unavailability
from algorithms.TESolver import TESolver
from algorithms.FFCSolver import FFCSolver
from algorithms.TEAVARSolver import TEAVARSolver
from algorithms.TUFTTEParameterSolver import TUFTTEParameterSolver, Availability
from algorithms.TUFTTEPredictSolver import TUFTTEPredictSolver
from algorithms.DoteSolver import DoteSolver, NeuralNetworkMaxUtil

import numpy as np
import torch
from collections import defaultdict

def compare_algorithms(num_train_samples=1000, num_test_samples=30, 
                       hist_len=12, demand_scale=0.3):
    """
    Compare TUFTTE (Parameter Learning) with baseline algorithms.
    
    Args:
        num_train_samples: Number of training TMs to use
        num_test_samples: Number of test TMs to evaluate
        hist_len: History length for prediction
        demand_scale: Demand scaling factor
    """
    print("="*70)
    print("Algorithm Comparison: TUFTTE (Parameter) vs Baselines")
    print("="*70)
    
    # Setup network
    print("\n1. Loading network...")
    network = parse_topology("GEANT")
    parse_histories(network)  # Load all available files
    network.reduce_data(num_train_samples, num_test_samples)  # Limit both train and test
    parse_tunnels(network, k=8)
    network.tunnel_type = "int"
    network.set_scale(demand_scale)
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
    
    print(f"\nNetwork Configuration:")
    print(f"  Topology: {network.name}")
    print(f"  Demand Scale: {demand_scale}")
    print(f"  Train TMs: {len(network.train_hists._tms)}")
    print(f"  Test TMs: {len(network.test_hists._tms)}")
    print(f"  Scenarios: {len(network.scenarios)}")
    
    results = {}
    
    # ========================
    # Prediction-based methods
    # ========================
    print("\n" + "-"*70)
    print("2. Testing Prediction-Based Methods")
    print("-"*70)
    
    # Get predictions using Random Forest
    # predicted_tms = predict_traffic_matrix(
    #     network.train_hists._tms, 
    #     network.test_hists._tms, 
    #     hist_len, 
    #     method=RF
    # )
    
    # for method in ["MaxMin", "MLU", "FFC", "TEAVAR"]:
    #     print(f"\n  Testing {method}...")
    #     network.clear_sol()
        
    #     for tm in predicted_tms:
    #         network.set_demand_amount(tm)
    #         lp = GurobiSolver()
            
    #         if method == "MaxMin":
    #             solver = TESolver(lp, network)
    #             solver.solve(obj="MaxMin")
    #         elif method == "MLU":
    #             solver = TESolver(lp, network)
    #             solver.solve(obj="MLU")
    #         elif method == "FFC":
    #             solver = FFCSolver(lp, network, 1)
    #             solver.solve()
    #         elif method == "TEAVAR":
    #             solver = TEAVARSolver(lp, network)
    #             solver.solve()
            
    #         sol = [tunnel.v_flow_value for tunnel in network.solutions.tunnels]
    #         network.add_sol(sol)
        
    #     _, availability = calculate_risk(network, hist_len)
    #     results[method] = {
    #         'availability_mean': np.mean(availability),
    #         'availability_std': np.std(availability),
    #         'availability_min': np.min(availability),
    #     }
    #     print(f"    Availability: {np.mean(availability):.6f} ± {np.std(availability):.6f}")
    #     network.clear_sol()
    
    # ========================
    # TUFTTE (Parameter Learning)
    # ========================
    print("\n" + "-"*70)
    print("3. Testing TUFTTE (Parameter Learning)")
    print("-"*70)
    
    network.clear_sol()
    solver = TUFTTEParameterSolver(network, hist_len=hist_len, type=Availability, suffix="_quick")
    solver.solve()
    
    _, availability = calculate_risk(network, hist_len)
    results["TUFTTE-Param"] = {
        'availability_mean': np.mean(availability),
        'availability_std': np.std(availability),
        'availability_min': np.min(availability),
    }
    print(f"  Availability: {np.mean(availability):.6f} ± {np.std(availability):.6f}")

    # ========================
    # TUFTTE (Original: Predict Demand)
    # ========================
    print("\n" + "-"*70)
    print("4. Testing TUFTTE (Original: Predict Demand)")
    print("-"*70)

    network.clear_sol()
    solver = TUFTTEPredictSolver(network, hist_len=hist_len, type=Availability, suffix="_quick")
    solver.solve()

    _, availability = calculate_risk(network, hist_len)
    results["TUFTTE-Orig"] = {
        'availability_mean': np.mean(availability),
        'availability_std': np.std(availability),
        'availability_min': np.min(availability),
    }
    print(f"  Availability: {np.mean(availability):.6f} ± {np.std(availability):.6f}")
    
    # ========================
    # DOTE
    # ========================
    print("\n" + "-"*70)
    print("5. Testing DOTE")
    print("-"*70)

    network.clear_sol()
    solver = DoteSolver(network, hist_len=hist_len, function="MAXUTIL")
    solver.solve()

    _, availability = calculate_risk(network, hist_len)
    results["DOTE"] = {
        'availability_mean': np.mean(availability),
        'availability_std': np.std(availability),
        'availability_min': np.min(availability),
    }
    print(f"  Availability: {np.mean(availability):.6f} ± {np.std(availability):.6f}")
    
    # ========================
    # Summary
    # ========================
    import sys
    sys.stdout.flush()  # 确保tqdm进度条输出已清理，避免与后续打印混乱

    print("\n" + "="*70)
    print("6. Results Summary")
    print("="*70)
    print(f"\nDemand Scale: {demand_scale}")
    print(f"\n{'Algorithm':<20} {'Mean Avail.':<15} {'Std':<12} {'Min':<12}")
    print("-"*60)
    
    # Sort by availability (descending)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['availability_mean'], reverse=True)
    
    for alg, metrics in sorted_results:
        print(f"{alg:<20} {metrics['availability_mean']:.6f}       {metrics['availability_std']:.6f}     {metrics['availability_min']:.6f}")
    
    # Find best
    best_alg = sorted_results[0][0]
    print(f"\n[Best] Algorithm: {best_alg} (Availability: {sorted_results[0][1]['availability_mean']:.6f})")
    
    # Check if TUFTTE-Param is best
    tuftte_rank = [i for i, (alg, _) in enumerate(sorted_results) if alg == "TUFTTE-Param"][0] + 1
    print(f"[Rank] TUFTTE-Param: {tuftte_rank}/{len(results)}")
    
    return results

def compare_across_scales(scales=None, **kwargs):
    """Compare algorithms across different demand scales."""
    if scales is None:
        scales = [0.1, 0.2, 0.3, 0.4]
    
    all_results = {}
    for scale in scales:
        print(f"\n{'#'*70}")
        print(f"# Demand Scale: {scale}")
        print(f"{'#'*70}")
        all_results[scale] = compare_algorithms(demand_scale=scale, **kwargs)
    
    # Summary table
    print("\n" + "="*70)
    print("FINAL SUMMARY: Availability Across Scales")
    print("="*70)
    
    algorithms = list(all_results[scales[0]].keys())
    
    print(f"\n{'Algorithm':<20}", end="")
    for scale in scales:
        print(f"Scale={scale:<8}", end="")
    print()
    print("-"*70)
    
    for alg in algorithms:
        print(f"{alg:<20}", end="")
        for scale in scales:
            avail = all_results[scale][alg]['availability_mean']
            print(f"{avail:.6f}   ", end="")
        print()
    
    return all_results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare TE algorithms")
    parser.add_argument("--scale", type=float, default=0.3, help="Demand scale")
    parser.add_argument("--multi-scale", action="store_true", help="Test multiple scales")
    parser.add_argument("--train-samples", type=int, default=1000, help="Number of train samples (default: 1000, matching benchmark)")
    parser.add_argument("--test-samples", type=int, default=300, help="Number of test samples (default: 30, for quick testing)")
    
    args = parser.parse_args()
    
    if args.multi_scale:
        compare_across_scales(
            scales=[0.1, 0.2, 0.3, 0.4],
            num_train_samples=args.train_samples,
            num_test_samples=args.test_samples
        )
    else:
        compare_algorithms(
            num_train_samples=args.train_samples,
            num_test_samples=args.test_samples,
            demand_scale=args.scale
        )
