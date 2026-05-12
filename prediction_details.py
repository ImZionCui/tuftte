"""
Details of prediction methods. (Figure 7, Figure 8, Figure 9)
"""

from utils.NetworkParser import *
from utils.scenario import scenarios_with_k_failed_links
from utils.prediction import predict_traffic_matrix, ALLMETHOD
from algorithms.TUFTTEParameterSolver import TUFTTEParameterSolver

import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict

def compute_MSE(topology, num_dms_for_train=None, num_dms_for_test=None, hist_len=12, plot=False):
    """
    Compute the MSE loss of distinct prediction methods. (Figure 7)

    parameters:
        topology(str): the name of topology;
        num_dms_for_train(int): the number of demand matrices for training/predicting;
        num_dms_for_test(int): the number of demand matrices for testing;
        hist_len(int): the number of historical demand matrices in a training instance;
        plot(bool): whether to plot the figure.
    """

    network = parse_topology(topology)
    parse_histories(network)
    network.reduce_data(num_dms_for_train, num_dms_for_test)
    real_tms = network.test_hists._tms[hist_len:]
    mse = defaultdict(list)
    for method in ALLMETHOD:
        predicted_tms = predict_traffic_matrix(network.train_hists._tms, network.test_hists._tms, hist_len, method)
        for i, tm in enumerate(predicted_tms):
            mse[method].append(np.mean((tm - real_tms[i]) ** 2))
    
    parse_tunnels(network, k=8)
    network.prepare_solution_format()
    scenarios_all = scenarios_with_k_failed_links(int(len(network.edges)/2), 1)
    network.set_scenario(scenarios_all)
    solver = TUFTTESolver(network, hist_len=hist_len)
    predicted_tms = solver.output_prediction()
    for i, tm in enumerate(predicted_tms):
        tm_cpu = tm[0].cpu() if torch.is_tensor(tm[0]) else tm[0]
        mse["TUFTTE"].append(torch.mean((torch.from_numpy(real_tms[i]) - tm_cpu) ** 2).item())

    if plot:
        fontsize = 20
        _, ax = plt.subplots()
        labels = []
        x_ticks = []
        for i, method in enumerate(mse.keys()):
            x_ticks.append(i)
            if method == 'AVG+STD':
                labels.append('AVG+\nSTD') # this string is too long to be split.
            else:
                labels.append(method)
            plt.boxplot(mse[method], positions=[i], widths=0.5, showfliers=False)

        ax.set_ylabel("MSE loss", fontsize=fontsize)
        ax.set_xticks(x_ticks)
        ax.tick_params(labelsize=fontsize)
        ax.set_xticklabels(labels)
        plt.savefig("plot.pdf", bbox_inches='tight')

def check_pos_neg(topology, num_dms_for_train=None, num_dms_for_test=None, hist_len=12, plot=False):
    """
    Compute the positive and negative bias of distinct prediction methods. (Figure 8)

    parameters:
        topology(str): the name of topology;
        num_dms_for_train(int): the number of demand matrices for training/predicting;
        num_dms_for_test(int): the number of demand matrices for testing;
        hist_len(int): the number of historical demand matrices in a training instance;
        plot(bool): whether to plot the figure.
    """

    network = parse_topology(topology)
    parse_histories(network)
    network.reduce_data(num_dms_for_train, num_dms_for_test)
    real_tms = network.test_hists._tms[hist_len:]
    positive = defaultdict(list)
    negative = defaultdict(list)
    for method in ALLMETHOD:
        predicted_tms = predict_traffic_matrix(network.train_hists._tms, network.test_hists._tms, hist_len, method)
        for i, tm in enumerate(predicted_tms):
            pos = 0
            neg = 0
            for j, d in enumerate(tm):
                bias = d - real_tms[i][j]
                if bias > 0:
                    pos += bias
                else:
                    neg += bias

            positive[method].append(pos)
            negative[method].append(neg)
    
    parse_tunnels(network, k=8)
    network.prepare_solution_format()
    scenarios_all = scenarios_with_k_failed_links(int(len(network.edges)/2), 1)
    network.set_scenario(scenarios_all)
    solver = TUFTTEParameterSolver(network, hist_len=hist_len)
    predicted_tms = solver.output_prediction()
    for i, tm in enumerate(predicted_tms):
        pos = 0
        neg = 0
        tm_cpu = tm[0].cpu() if torch.is_tensor(tm[0]) else tm[0]
        for j, d in enumerate(tm_cpu):
            bias = d.item() - real_tms[i][j]
            if bias > 0:
                pos += bias
            else:
                neg += bias

        positive["TUFTTE"].append(pos)
        negative["TUFTTE"].append(neg)
    
    if plot:
        fontsize = 20
        bar_width = 0.45
        bar_labels = np.arange(len(positive))
        labels = []
        for method in positive.keys():
            if method == 'AVG+STD':
                labels.append('AVG+\nSTD') # this string is too long to be split.
            else:
                labels.append(method)
        data1 = [np.mean(pos) for pos in positive.values()]
        data2 = [np.mean(neg) for neg in negative.values()]
        plt.figure()
        plt.bar(bar_labels, data1, width=bar_width, label="positive", hatch='O')
        plt.bar(bar_labels, data2, width=bar_width, label="negative", hatch='*')
        plt.ylabel("Bias", fontsize=fontsize)
        plt.xticks(bar_labels, labels)
        plt.tick_params(axis='both', which='major', labelsize=fontsize)
        plt.legend(fontsize=fontsize)
        plt.savefig("plot.pdf", bbox_inches='tight')

def watch_pos_neg_variation(topology, num_dms_for_train=None, hist_len=12, demand_scale=1, plot=False):
    """
    Track the variation of the positive and negative bias when TUFTTE is training. (Figure 9)

    parameters:
        topology(str): the name of topology;
        num_dms_for_train(int): the number of demand matrices for training;
        hist_len(int): the number of historical demand matrices in a training instance;
        demand_scale(float): the demand scale factor used to multiply the demand matrix;
        plot(bool): whether to plot the figure.
    """

    network = parse_topology(topology)
    parse_histories(network, demand_scale)
    network.reduce_data(num_dms_for_train)
    parse_tunnels(network, k=8)
    network.prepare_solution_format()
    scenarios_all = scenarios_with_k_failed_links(int(len(network.edges)/2), 1)
    network.set_scenario(scenarios_all)
    solver = TUFTTEParameterSolver(network, hist_len=hist_len)
    positive, negative = solver.fake_train()
    print(positive)
    print(negative)

    #plot:是否需要可视化
    if plot:
        negative = [-neg for neg in negative] # Absolute value
        fontsize = 20
        length = len(positive)
        plt.plot(positive, label="positive", marker='o', markevery=length//20)
        plt.plot(negative, label="negative", marker='s', markevery=length//20)
        plt.xlabel("Epochs", fontsize=fontsize)
        plt.ylabel("Absolute value of bias", fontsize=fontsize)
        plt.tick_params(axis='both', which='major', labelsize=fontsize)
        plt.legend(fontsize=fontsize)
        plt.savefig("plot.pdf", bbox_inches='tight')