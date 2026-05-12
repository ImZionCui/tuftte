"""
Availability vs. demand scales for TUFTTE and various TE schemes. (Figure 6)
"""

from utils.NetworkParser import *
from utils.prediction import predict_traffic_matrix, RF
from utils.scenario import subScenarios
from utils.GurobiSolver import GurobiSolver
from utils.riskMetric import calculate_risk
from algorithms.TESolver import TESolver
from algorithms.FFCSolver import FFCSolver
from algorithms.TEAVARSolver import TEAVARSolver
from algorithms.DoteSolver import DoteSolver
from algorithms.TUFTTEParameterSolver import TUFTTEParameterSolver, Availability
from algorithms.TUFTTEPredictSolver import TUFTTEPredictSolver

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

PREDICTION_BASED_METHODS = ["MaxMin", "MLU", "FFC", "TEAVAR"]
DIRECT_OPTIMIZATION = ["DOTE", "TUFTTE-Param", "TUFTTE-Orig"]

def availability_plot(topology, num_dms_for_train=None, num_dms_for_test=None, cutoff=1e-6, hist_len=12, plot=False, start=0.1, step=0.01, stop=0.4):
    """
    parameters:
        topology(str): the name of topology;
        num_dms_for_train(int): the number of demand matrices for training/predicting;
        num_dms_for_test(int): the number of demand matrices for testing;
        cutoff(float): consider the scenarios whose probability exceeds cutoff;
        hist_len(int): the number of historical demand matrices in a training instance;
        plot(bool): whether to plot the figure;
        start(float): the smallest demand scale factor;
        step(float): the distance between two adjacent demand scale factors;
        stop(float): the biggest demand scale factor.
    """

    network = parse_topology(topology)
    parse_histories(network)
    network.reduce_data(num_dms_for_train, num_dms_for_test)
    parse_tunnels(network, k=8)
    network.prepare_solution_format()
    prob_failure = []
    edge_included = []
    # edge and its inverse edge fail together
    for edge in network.edges:
        if set(edge) in edge_included:
            continue

        prob_failure.append(network.edges[edge].prob_failure)
        edge_included.append(set(edge))
        
    scenarios_all = subScenarios(prob_failure, cutoff)
    network.set_scenario(scenarios_all)
    method_loss = defaultdict(list)
    scales = [start]
    while start < stop:
        start += step
        scales.append(start)

    print("\n=== Demand Scales ===")
    print(", ".join([f"{s:.3f}" for s in scales]))

    predicted_tms = predict_traffic_matrix(network.train_hists._tms, network.test_hists._tms, hist_len, method=RF)
    for method in PREDICTION_BASED_METHODS:
        print(f"\n[Prediction-based] {method}")
        for demand_scale in scales:
            network.set_scale(demand_scale)
            for tm in predicted_tms:
                network.set_demand_amount(tm)
                lp = GurobiSolver()
                if method == "MaxMin":
                    solver = TESolver(lp, network)
                    solver.solve(obj=method)
                elif method == "MLU":
                    solver = TESolver(lp, network)
                    solver.solve(obj=method)
                elif method == "FFC":
                    solver = FFCSolver(lp, network, 1)
                    solver.solve()
                elif method == "TEAVAR":
                    solver = TEAVARSolver(lp, network)
                    solver.solve()
                else:
                    print(f"Method {method} is not defined!")
                    return

                sol = [tunnel.v_flow_value for tunnel in network.solutions.tunnels]
                network.add_sol(sol)

            _, availability = calculate_risk(network, hist_len)
            network.clear_sol()
            avg_availability = np.mean(availability)
            method_loss[method].append(avg_availability)
            print(f"  scale={demand_scale:.3f} -> availability={avg_availability:.8f}")

    # Use a fixed pre-trained scale/suffix for TUFTTE models
    train_scale = 15.0
    train_suffix = "_quick"

    for method in DIRECT_OPTIMIZATION:
        print(f"\n[Direct-optimization] {method}")
        if method == "DOTE":
            solver = DoteSolver(network)
        elif method == "TUFTTE-Param":
            network.set_scale(train_scale)
            solver = TUFTTEParameterSolver(network, type=Availability, suffix=train_suffix)
        elif method == "TUFTTE-Orig":
            network.set_scale(train_scale)
            solver = TUFTTEPredictSolver(network, type=Availability, suffix=train_suffix)
        else:
            print(f"Method {method} is not defined!")
            return

        solver.solve()
        for demand_scale in scales:
            network.set_scale(demand_scale)
            _, availability = calculate_risk(network, hist_len)
            avg_availability = np.mean(availability)
            method_loss[method].append(avg_availability)
            print(f"  scale={demand_scale:.3f} -> availability={avg_availability:.8f}")

        network.clear_sol()
        
    print(method_loss)

    # 打印按 scale 汇总的 availability 表
    print("\n=== Availability by Scale (Summary) ===")
    header = f"{'Scale':<10}" + "".join([f"{m:<16}" for m in method_loss.keys()])
    print(header)
    print("-" * len(header))
    for i, scale in enumerate(scales):
        row = f"{scale:<10.3f}"
        for method in method_loss.keys():
            row += f"{method_loss[method][i]:<16.8f}"
        print(row)

    if plot:
        marker_styles = ['o', 's', '^', 'v', '<', '>', 'p', 'h', 'D', '*', '+', 'x']
        fontsize = 20
        for i, method in enumerate(method_loss):
            plt.plot(scales, method_loss[method], marker=marker_styles[i], label = method)
            
        plt.xlabel("Demand Scale", fontsize=fontsize)
        plt.ylabel("Availability", fontsize=fontsize)
        plt.tick_params(axis='both', which='major', labelsize=fontsize)
        # plt.ylim((0.999, 1.0001))
        plt.ylim((0.999, 1.0001))
        plt.legend()
        plt.savefig("plot_availability.pdf", bbox_inches='tight')
        
    # availability_vals = {alg : [] for alg in algorithms}
    # for topo in topologies:
    #     print("Topology: %s" % topo)
    #     network = parse_topology(topo, use_weibull=True)
    #     parse_tunnels(network, paths=paths, k=k)
    #     parse_demands(network)
    #     initialize_weights(network)
    #     prob_failure = []
    #     edge_included = []
    #     # edge and its inverse edge fail together
    #     for edge in network.edges:
    #         if set(edge) in edge_included:
    #             continue

    #         prob_failure.append(network.edges[edge].prob_failure)
    #         edge_included.append(set(edge))
            
    #     scenarios_all = subScenarios(prob_failure, cutoff)
    #     scenarios_probs_all = [s.prob for s in scenarios_all]
    #     network.set_scenario(scenarios_all)

    #     for alg in algorithms:
    #         print("Using %s" % alg)
    #         availabilities = []

    #         for s in scales:
    #             print("Scale: %.2f" % s)
    #             network.set_scale(s)
    #             lp = GurobiSolver()
    #             if alg.startswith("FFC"):
    #                 k = int(alg[-1])
    #                 solver = FFCSolver(lp, network, k)
    #                 solver.solve()
    #             elif alg == "TEAVAR":
    #                 solver = TEAVARSolver(lp, network, scenarios_probs_all[0] - 0.01)
    #                 solver.solve()
    #             elif alg == "SMORE":
    #                 solver = TESolver(lp, network)
    #                 solver.solve(obj='SMORE')
    #             elif alg == "MaxMin":
    #                 solver = TESolver(lp, network)
    #                 solver.solve(obj='MaxMin')

    #             loss = calculateLossReallocation(network)
    #             # availabilities.append(np.sum(np.multiply(loss, scenarios_probs_all)))
    #             availabilities.append(sum(scenarios_probs_all[i] for i in range(len(loss)) if loss[i] <= sla))

    #         availability_vals[alg].append(availabilities)
    #         print(availabilities)

    # if plot:
    #     for alg in availability_vals:
    #         plt.plot(scales, np.mean(availability_vals[alg], axis=0))
        
    #     plt.xlabel("Demand Scale")
    #     plt.ylabel("Availability")
    #     plt.ylim((0.95, 1.0001))
    #     plt.legend(algorithms)
    #     plt.savefig("plot.pdf")
    #     plt.show()