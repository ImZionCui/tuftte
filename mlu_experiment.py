"""
Maximum Link Utilization (MLU) comparison for distinct methods.
"""

from utils.NetworkParser import *
from utils.prediction import predict_traffic_matrix, RF
from utils.scenario import scenarios_with_k_failed_links
from utils.GurobiSolver import GurobiSolver
from algorithms.TESolver import TESolver
from algorithms.FFCSolver import FFCSolver
from algorithms.TEAVARSolver import TEAVARSolver
from algorithms.DoteSolver import DoteSolver
from algorithms.TUFTTEPredictSolver import TUFTTEPredictSolver
from algorithms.TUFTTEParameterSolver import TUFTTEParameterSolver

import matplotlib.pyplot as plt
import numpy as np
import torch

DemandLoss = "D"
Availability = "A"

PREDICTION_BASED_METHODS = ["MaxMin", "MLU", "FFC", "TEAVAR"]
DIRECT_OPTIMIZATION = ["TUFTTE-Predict", "TUFTTE-Param"]

def calculate_mlu(network, solution, debug=False, debug_label=""):
    """
    计算给定路由方案下的最大链路利用率
    
    parameters:
        network: 网络拓扑
        solution: 路由方案(隧道流量分配)，可以是torch.Tensor或numpy数组
        debug: 是否打印所有链路的利用率
        
    returns:
        mlu: 最大链路利用率 (0-1之间，或大于1表示超载)
        total_flow: 所有链路流量总和
        active_links: 有流量的链路数量
    """
    import torch
    
    # 转换为numpy
    if isinstance(solution, torch.Tensor):
        solution = solution.detach().cpu().numpy()
    
    max_utilization = 0.0
    edge_utils = []
    total_flow = 0.0
    active_links = 0
    
    for edge in network.edges.values():
        edge_flow = sum(float(solution[t.id]) for t in edge.tunnels)
        utilization = edge_flow / edge.capacity if edge.capacity > 0 else 0
        edge_utils.append((edge.e, utilization, edge_flow, edge.capacity))
        max_utilization = max(max_utilization, utilization)
        total_flow += edge_flow
        if edge_flow > 1e-3:  # 流量大于0.001 Mbps认为活跃
            active_links += 1
    
    if debug:
        # 打印所有边的利用率，按利用率排序
        edge_utils.sort(key=lambda x: x[1], reverse=True)
        label = f" [{debug_label}]" if debug_label else ""
        print(f"\n[DEBUG]{label} All edges utilization:")
        print(f"  Total flow across all links: {total_flow:.2f} Mbps")
        print(f"  Active links (flow > 0.001): {active_links}/{len(network.edges)}")
        print(f"  Average flow per active link: {total_flow/active_links:.2f} Mbps")
        for i, (edge_name, util, flow, capacity) in enumerate(edge_utils):
            print(f"  {i+1}. Edge {edge_name}: flow={flow:.2f}, capacity={capacity:.2f}, util={util:.4f}")
    
    return max_utilization, total_flow, active_links

def mlu_comparison_expr(topology, num_dms_for_train=None, num_dms_for_test=None, K=1, hist_len=12, demand_scale=1, plot=False):
    """
    对比不同方法的最大链路利用率
    
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
    method_mlu = {}
    predicted_tms = predict_traffic_matrix(network.train_hists._tms, network.test_hists._tms, hist_len, method=RF)
    
    # 预测类方法
    for method in PREDICTION_BASED_METHODS:
        mlu_values = []
        total_flows = []
        active_links_list = []
        for i, tm in enumerate(predicted_tms):
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
            # 对对比方案中的 MLU 方法，在第一个样本打印链路利用率明细
            should_debug = (method == "MLU" and i == 0)
            if should_debug:
                print("\n[DEBUG] Trigger edge-util print for MLU sample 0")
            mlu, total_flow, active_links = calculate_mlu(
                network,
                sol,
                debug=should_debug,
                debug_label=f"{method} sample {i}"
            )
            mlu_values.append(mlu)
            total_flows.append(total_flow)
            active_links_list.append(active_links)
            network.add_sol(sol)

        network.clear_sol()
        method_mlu[method] = mlu_values
        # 打印该方法的流量统计
        avg_flow = np.mean(total_flows)
        avg_active = np.mean(active_links_list)
        print(f"[{method}] 平均总流量={avg_flow:.2f} Mbps, 平均活跃链路数={avg_active:.1f}/30")

    # 直接优化方法（TUFTTE系列）
    for method in DIRECT_OPTIMIZATION:
        if method == "TUFTTE-Predict":
            # solver = TUFTTEPredictSolver(network, hist_len=hist_len, type=DemandLoss)
            solver = TUFTTEPredictSolver(network, hist_len=hist_len, type=Availability, suffix="_quick")
        elif method == "TUFTTE-Param":
            # solver = TUFTTEParameterSolver(network, hist_len=hist_len, type=DemandLoss)
            solver = TUFTTEParameterSolver(network, hist_len=hist_len, type=Availability, suffix="_quick")
        else:
            print(f"Method {method} is not defined!")
            return

        solver.solve()
        
        # 计算每个测试样本的MLU
        mlu_values = []
        total_flows = []
        active_links_list = []
        if hasattr(network.solutions, 'val') and network.solutions.val:
            for i, sol in enumerate(network.solutions.val):
                # 第一个样本打印详细信息
                should_debug = (method=="TUFTTE-Param" and i==0)
                if should_debug:
                    print("\n[DEBUG] Trigger edge-util print for TUFTTE-Param sample 0")
                mlu, total_flow, active_links = calculate_mlu(
                    network,
                    sol,
                    debug=should_debug,
                    debug_label=f"{method} sample {i}"
                )
                mlu_values.append(mlu)
                total_flows.append(total_flow)
                active_links_list.append(active_links)
        
        network.clear_sol()
        method_mlu[method] = mlu_values
        # 打印该方法的流量统计
        if total_flows:
            avg_flow = np.mean(total_flows)
            avg_active = np.mean(active_links_list)
            print(f"[{method}] 平均总流量={avg_flow:.2f} Mbps, 平均活跃链路数={avg_active:.1f}/30")

    # 打印统计信息
    print("\n=== MLU Comparison ===")
    for method in method_mlu.keys():
        avg_mlu = np.mean(method_mlu[method])
        median_mlu = np.median(method_mlu[method])
        max_mlu = np.max(method_mlu[method])
        min_mlu = np.min(method_mlu[method])
        print(f"{method:15s}: 平均={avg_mlu:.4f}, 中位数={median_mlu:.4f}, 最大={max_mlu:.4f}, 最小={min_mlu:.4f}")
    
    # 对比TUFTTE变体
    if "TUFTTE-Predict" in method_mlu and "TUFTTE-Param" in method_mlu:
        print("\n=== TUFTTE Variants MLU Comparison ===")
        diff_sum = 0.0
        param_wins = 0
        predict_wins = 0
        total = len(method_mlu["TUFTTE-Predict"])
        
        for i in range(total):
            diff = method_mlu["TUFTTE-Param"][i] - method_mlu["TUFTTE-Predict"][i]
            diff_sum += diff
            if diff < -0.001:  # Param MLU更低（更好）
                param_wins += 1
            elif diff > 0.001:  # Predict MLU更低（更好）
                predict_wins += 1
            
            if i < 5:
                print(f"Sample {i}: Param={method_mlu['TUFTTE-Param'][i]:.4f}, Predict={method_mlu['TUFTTE-Predict'][i]:.4f}, Diff={diff:.4f}")
        
        print(f"\n总样本数: {total}")
        print(f"平均差异 (Param - Predict): {diff_sum/total:.4f}")
        print(f"Param更优(MLU更低): {param_wins} ({param_wins/total*100:.1f}%)")
        print(f"Predict更优(MLU更低): {predict_wins} ({predict_wins/total*100:.1f}%)")
        print(f"相近: {total - param_wins - predict_wins} ({(total - param_wins - predict_wins)/total*100:.1f}%)")

    if plot:
        fontsize = 20
        marker_styles = ['o', 's', '^', 'v', '<', '>', 'p', 'h', 'D', '*', '+', 'x']
        cdf = np.arange(len(predicted_tms)) / (len(predicted_tms) - 1)
        for i, method in enumerate(method_mlu.keys()):
            data = np.sort(method_mlu[method])
            plt.plot(data, cdf, label=method, marker=marker_styles[i], markevery=len(data)//15)

        plt.xlabel("Maximum Link Utilization", fontsize=fontsize)
        plt.ylabel("CDF", fontsize=fontsize)
        plt.tick_params(axis='both', which='major', labelsize=fontsize)
        plt.xlim((0, 1.2))
        plt.legend()
        plt.savefig('plot_mlu_experiment.pdf', bbox_inches='tight')
        print("\n图表已保存到 plot_mlu_experiment.pdf")
