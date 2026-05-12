"""
Demand loss for distinct methods. (Figure 4, Figure 5, Table 2)
"""

from utils.NetworkParser import *
from utils.prediction import predict_traffic_matrix, RF
from utils.scenario import scenarios_with_k_failed_links
from utils.GurobiSolver import GurobiSolver
from utils.riskMetric import calculate_risk
from algorithms.TESolver import TESolver
from algorithms.FFCSolver import FFCSolver
from algorithms.TEAVARSolver import TEAVARSolver
from algorithms.DoteSolver import DoteSolver
from algorithms.TUFTTESolver_old import TUFTTESolver, Dsolver
from algorithms.TUFTTEPredictSolver import TUFTTEPredictSolver
from algorithms.TUFTTEParameterSolver import TUFTTEParameterSolver

import matplotlib.pyplot as plt
import numpy as np

DemandLoss = "D"
Availability = "A"

# PREDICTION_BASED_METHODS = ["MaxMin", "MLU", "FFC", "TEAVAR"]
# DIRECT_OPTIMIZATION = ["DOTE", "TUFTTE-Predict", "TUFTTE-Param"]
DIRECT_OPTIMIZATION = ["DOTE"]

def demand_loss_expr(topology, num_dms_for_train=None, num_dms_for_test=None, K=1, hist_len=12, demand_scale=1, plot=False):
    """
    parameters:
        topology(str): the name of topology;
        num_dms_for_train(int): the number of demand matrices for training/predicting;
        num_dms_for_test(int): the number of demand matrices for testing;
        K(int): consider the scenarios with K failed links;
        hist_len(int): the number of historical demand matrices in a training instance;
        demand_scale(float): the demand scale factor used to multiply the demand matrix;
        plot(bool): whether to plot the figure.
    """

    network = parse_topology(topology)
    parse_histories(network, demand_scale)
    network.reduce_data(num_dms_for_train, num_dms_for_test)
    parse_tunnels(network, k=8)
    network.prepare_solution_format()
    scenarios_all = scenarios_with_k_failed_links(int(len(network.edges)/2), K)
    network.set_scenario(scenarios_all)
    method_loss = {}
    predicted_tms = predict_traffic_matrix(network.train_hists._tms, network.test_hists._tms, hist_len, method=RF)
    
    
    
    # for method in PREDICTION_BASED_METHODS:
    #     for tm in predicted_tms:
    #         network.set_demand_amount(tm)
    #         lp = GurobiSolver()
    #         if method == "MaxMin":
    #             solver = TESolver(lp, network)
    #             solver.solve(obj=method)
    #         elif method == "MLU":
    #             solver = TESolver(lp, network)
    #             solver.solve(obj=method)
    #         elif method == "FFC":
    #             solver = FFCSolver(lp, network, K)
    #             solver.solve()
    #         elif method == "TEAVAR":
    #             solver = TEAVARSolver(lp, network)
    #             solver.solve()
    #         else:
    #             print(f"Method {method} is not defined!")
    #             return

    #         sol = [tunnel.v_flow_value for tunnel in network.solutions.tunnels]
    #         network.add_sol(sol)

    #     demand_loss, _ = calculate_risk(network, hist_len)
    #     network.clear_sol()
    #     method_loss[method] = demand_loss



    for method in DIRECT_OPTIMIZATION:
        if method == "DOTE":
            solver = DoteSolver(network)
        elif method == "TUFTTE-Predict":
            solver = TUFTTEPredictSolver(network, hist_len=hist_len, type=DemandLoss)
        elif method == "TUFTTE-Param":
            solver = TUFTTEParameterSolver(network, hist_len=hist_len, type=DemandLoss)
        else:
            print(f"Method {method} is not defined!")
            return

        solver.solve()
        demand_loss, _ = calculate_risk(network, hist_len)
        network.clear_sol()
        method_loss[method] = demand_loss

    # 对比TUFTTE变体与FFC
    for tuftte_method in ["TUFTTE-Predict", "TUFTTE-Param"]:
        loss_reduction = 0.0
        max_loss = 0.0
        for i, loss in enumerate(method_loss[tuftte_method]):
            if loss > 0.0:
                loss_reduction += (method_loss["FFC"][i] - loss) / method_loss["FFC"][i]
                max_loss = max(max_loss, (method_loss["FFC"][i] - loss) / method_loss["FFC"][i])
            elif method_loss["FFC"][i] > 0.0:
                loss_reduction += 1
        print(f"{tuftte_method}'s demand loss is less than FFC's averagely by ", loss_reduction / len(method_loss[tuftte_method]))
        print(f"{tuftte_method} maximum discrepancy: ", max_loss)
    
    # 对比TUFTTE-Predict和TUFTTE-Param
    print("\n=== TUFTTE Variants Comparison ===")
    diff_sum = 0.0
    abs_diff_sum = 0.0
    max_diff = 0.0
    max_diff_idx = -1
    param_wins = 0  # Param损失更低的样本数
    predict_wins = 0  # Predict损失更低的样本数
    
    for i in range(len(method_loss["TUFTTE-Predict"])):
        diff = method_loss["TUFTTE-Param"][i] - method_loss["TUFTTE-Predict"][i]
        diff_sum += diff
        abs_diff_sum += abs(diff)
        
        if abs(diff) > abs(max_diff):
            max_diff = diff
            max_diff_idx = i
        
        if diff < -0.01:  # Param更好
            param_wins += 1
        elif diff > 0.01:  # Predict更好
            predict_wins += 1
            
        if i < 5:  # 打印前几个样本
            print(f"Sample {i}: Param={method_loss['TUFTTE-Param'][i]:.2f}, Predict={method_loss['TUFTTE-Predict'][i]:.2f}, Diff={diff:.2f}")
    
    total_samples = len(method_loss['TUFTTE-Predict'])
    print(f"\n总样本数: {total_samples}")
    print(f"平均差异 (Param - Predict): {diff_sum / total_samples:.2f}")
    print(f"平均绝对差异: {abs_diff_sum / total_samples:.2f}")
    print(f"最大差异: {max_diff:.2f} (样本 {max_diff_idx})")
    print(f"Param更优的样本数: {param_wins} ({param_wins/total_samples*100:.1f}%)")
    print(f"Predict更优的样本数: {predict_wins} ({predict_wins/total_samples*100:.1f}%)")
    print(f"相近的样本数: {total_samples - param_wins - predict_wins} ({(total_samples - param_wins - predict_wins)/total_samples*100:.1f}%)")
    
    if plot:
        fontsize = 20
        marker_styles = ['o', 's', '^', 'v', '<', '>', 'p', 'h', 'D', '*', '+', 'x']
        cdf = np.arange(len(predicted_tms)) / (len(predicted_tms) - 1)
        for i, method in enumerate(method_loss.keys()):
            data = np.sort(method_loss[method])
            plt.plot(data, cdf, label=method, marker=marker_styles[i], markevery=len(data)//15)

        plt.xlabel("Demand loss (Mbps)", fontsize=fontsize)
        plt.ylabel("CDF", fontsize=fontsize)
        plt.tick_params(axis='both', which='major', labelsize=fontsize)
        plt.xlim((0, 10000))
        # plt.xlim((0,8000))
        plt.legend()
        plt.savefig('plot_dl_experiment.pdf', bbox_inches='tight')

def noise_expr(topology, num_dms_for_train=None, num_dms_for_test=None, K=1, hist_len=12, demand_scale=1, noise=0.3, plot=False):
    """
    parameters:
        topology(str): the name of topology;
        num_dms_for_train(int): the number of demand matrices for training/predicting;
        num_dms_for_test(int): the number of demand matrices for testing;
        K(int): consider the scenarios with K failed links;
        hist_len(int): the number of historical demand matrices in a training instance;
        demand_scale(float): the demand scale factor used to multiply the demand matrix;
        noise(float): the range of noise;
        plot(bool): whether to plot the figure.
    """

    network = parse_topology(topology)
    parse_histories(network, demand_scale)
    network.reduce_data(num_dms_for_train, num_dms_for_test)
    parse_tunnels(network, k=8)
    network.prepare_solution_format()
    scenarios_all = scenarios_with_k_failed_links(int(len(network.edges)/2), K)
    network.set_scenario(scenarios_all)
    method_loss = {}
    predicted_tms = predict_traffic_matrix(network.train_hists._tms, network.test_hists._tms, hist_len, method=RF)
    real_tms = network.test_hists._tms
    noisy_tms = make_noise(real_tms, noise)
    network.test_hists._tms = noisy_tms
    for method in PREDICTION_BASED_METHODS:
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
                solver = FFCSolver(lp, network, K)
                solver.solve()
            elif method == "TEAVAR":
                solver = TEAVARSolver(lp, network)
                solver.solve()
            else:
                print(f"Method {method} is not defined!")
                return

            sol = [tunnel.v_flow_value for tunnel in network.solutions.tunnels]
            network.add_sol(sol)

        demand_loss, _ = calculate_risk(network, hist_len)
        network.clear_sol()
        method_loss[method] = demand_loss

    for method in DIRECT_OPTIMIZATION:
        if method == "DOTE":
            solver = DoteSolver(network)
        elif method == "TUFTTE-Predict":
            solver = TUFTTEPredictSolver(network, hist_len=hist_len, type=DemandLoss)
        elif method == "TUFTTE-Param":
            solver = TUFTTEParameterSolver(network, hist_len=hist_len, type=DemandLoss)
        else:
            print(f"Method {method} is not defined!")
            return

        # train without noises, and compute network risks with noises
        network.test_hists._tms = real_tms
        solver.solve()
        network.test_hists._tms = noisy_tms
        demand_loss, _ = calculate_risk(network, hist_len)
        network.clear_sol()
        method_loss[method] = demand_loss

    # loss_reduction = 0.0
    # for i, loss in enumerate(method_loss["TUFTTE"]):
    #     if loss > 0.0:
    #         loss_reduction += (loss - method_loss["FFC"][i]) / loss
    #     elif method_loss["FFC"][i] > 0.0:
    #         loss_reduction += 1
    # print(loss_reduction / len(method_loss["TUFTTE"]))

    if plot:
        fontsize = 20
        marker_styles = ['o', 's', '^', 'v', '<', '>', 'p', 'h', 'D', '*', '+', 'x']
        cdf = np.arange(len(predicted_tms)) / (len(predicted_tms) - 1)
        for i, method in enumerate(method_loss.keys()):
            data = np.sort(method_loss[method])
            plt.plot(data, cdf, label=method, marker=marker_styles[i], markevery=len(data)//15)

        plt.xlabel("Demand loss (Mbps)", fontsize=fontsize)
        plt.ylabel("CDF", fontsize=fontsize)
        plt.tick_params(axis='both', which='major', labelsize=fontsize)
        plt.xlim((1000, 12000))
        # plt.xlim((0, 8000))
        plt.legend()
        plt.savefig('plot.pdf', bbox_inches='tight')

def noise_effect(topology, num_dms_for_train=None, num_dms_for_test=None, K=1, hist_len=12, demand_scale=1, noises=[0.1, 0.15, 0.2, 0.25, 0.3]):
    """
    parameters:
        topology(str): the name of topology;
        num_dms_for_train(int): the number of demand matrices for training/predicting;
        num_dms_for_test(int): the number of demand matrices for testing;
        K(int): consider the scenarios with K failed links;
        hist_len(int): the number of historical demand matrices in a training instance;
        demand_scale(float): the demand scale factor used to multiply the demand matrix;
        noises(list of `float`): list of concerning noises.
    """

    network = parse_topology(topology)
    parse_histories(network, demand_scale)
    network.reduce_data(num_dms_for_train, num_dms_for_test)
    parse_tunnels(network, k=8)
    network.prepare_solution_format()
    scenarios_all = scenarios_with_k_failed_links(int(len(network.edges)/2), K)
    network.set_scenario(scenarios_all)
    real_tms = network.test_hists._tms
    noisy_tms = {}
    for noise in noises:
        noisy_tms[noise] = make_noise(real_tms, noise)
    solver = TUFTTEParameterSolver(network, hist_len=hist_len, type=DemandLoss)
    # train without noises, and compute network risks with noises
    solver.solve()
    dl_without_noise, _ = calculate_risk(network, hist_len)
    noise_effect = []
    for noise in noises:
        network.test_hists._tms = noisy_tms[noise]
        dl_with_noise, _ = calculate_risk(network, hist_len)
        e = 0.0
        for i, dl in enumerate(dl_with_noise):
            if dl != 0.0:
                e += (dl - dl_without_noise[i]) / dl
        
        noise_effect.append(e / len(dl_without_noise))

    print(f"TUFTTE's increment in demand loss: {noise_effect}")
    network.clear_sol()
    predicted_tms = predict_traffic_matrix(network.train_hists._tms, real_tms, hist_len, method=RF)
    for tm in predicted_tms:
        network.set_demand_amount(tm)
        lp = GurobiSolver()
        solver = FFCSolver(lp, network, K)
        solver.solve()
        sol = [tunnel.v_flow_value for tunnel in network.solutions.tunnels]
        network.add_sol(sol)
        
    noise_effect = []
    for noise in noises:
        network.test_hists._tms = noisy_tms[noise]
        dl_with_noise, _ = calculate_risk(network, hist_len)
        e = 0.0
        for i, dl in enumerate(dl_with_noise):
            if dl != 0.0:
                e += (dl - dl_without_noise[i]) / dl
        
        noise_effect.append(e / len(dl_without_noise))

    print(f"FFC's increment in demand loss: {noise_effect}")
    
def make_noise(tms, noise):
    """
    parameters:
        tms(list of `list of `float``): a series of demand matrices;
        noise(float): the range of noise.

    returns:
        noisy_tms(list of `list of `float``): a series of demand matrices with noise.
    """
    return [[d *(1 + np.random.uniform(-noise, noise)) for d in tm] for tm in tms]
    