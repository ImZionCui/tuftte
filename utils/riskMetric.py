"""
This file includes how to evaluate the network risk.
"""

import numpy as np
import torch

def calculate_risk(network, hist_len=12):
    scenarios = network.scenarios
    solutions = network.solutions
    num_demands = len(solutions.val)
    demand_loss = np.zeros(num_demands)
    availability = np.zeros(num_demands)
    assert(len(solutions.val[0]) == len(network.tunnels))

    print("Computing loss...")
    for i, sol in enumerate(solutions.val):
        demand_matrix = network.test_hists._tms[i + hist_len]
        total_demand = sum(demand_matrix) * network.scale
        for s in scenarios:
            # compute amount of traffic on each tunnel
            routed = np.zeros_like(sol)
            for demand in network.demands.values():
                ratio = sum(sol[t.id] for t in demand.tunnels if t.pathstr not in s.failed_tunnels)
                if ratio == 0:
                    num_paths = len([t for t in demand.tunnels if t.pathstr not in s.failed_tunnels])
                    if num_paths == 0: continue
                    for t in demand.tunnels:
                        if t.pathstr not in s.failed_tunnels:
                            routed[t.id] = demand_matrix[demand.id] * network.scale / num_paths
                else:
                    ratio = demand_matrix[demand.id] * network.scale / ratio
                    for t in demand.tunnels:
                        if t.pathstr not in s.failed_tunnels:
                            routed[t.id] = sol[t.id] * ratio

            # compute congestion loss
            # eu = {}
            for edge in network.edges.values():
                edge_utilization = sum(routed[t.id] for t in edge.tunnels)
                if edge_utilization > edge.capacity:
                    tunnels = [t for t in edge.tunnels if t.pathstr not in s.failed_tunnels]
                    tunnels.sort(key=lambda t: len(t))
                    while edge_utilization > edge.capacity + 1e-3:
                        congestion_loss = min(routed[tunnels[-1].id], edge_utilization - edge.capacity)
                        edge_utilization -= congestion_loss
                        routed[tunnels[-1].id] -= congestion_loss
                        tunnels.pop()

                # eu[edge.e] = int(edge_utilization / edge.capacity * 100)
            
            traffic_loss = total_demand - sum(routed)
            if traffic_loss < 1e-3:
                availability[i] += s.prob
            elif traffic_loss > demand_loss[i]:
                # network.draw(eu)
                demand_loss[i] = traffic_loss

    return demand_loss, availability

def validate_demand_loss(network, solution, real_tm):
    scenarios = network.scenarios
    real_tm.requires_grad_()
    device = solution.device
    demand_loss = torch.tensor(0.0, device=device)
    for s in scenarios:
        # compute amount of traffic on each tunnel
        routed = torch.zeros_like(solution)
        for demand in network.demands.values():
            ratio = torch.sum(torch.tensor([solution[t.id] for t in demand.tunnels if t.pathstr not in s.failed_tunnels], device=device))
            if ratio.item() == 0:
                num_paths = len([t for t in demand.tunnels if t.pathstr not in s.failed_tunnels])
                if num_paths == 0: continue
                for t in demand.tunnels:
                    if t.pathstr not in s.failed_tunnels:
                        routed[t.id] = real_tm[demand.id] * torch.tensor(network.scale, device=device) / num_paths
            else:
                ratio = real_tm[demand.id] * torch.tensor(network.scale, device=device) / ratio
                for t in demand.tunnels:
                    if t.pathstr not in s.failed_tunnels:
                        routed[t.id] = solution[t.id] * ratio

        traffic_loss = torch.tensor(0.0, device=device)
        # compute congestion loss
        for edge in network.edges.values():
            edge_utilization = torch.tensor(0.0, device=device)
            for t in edge.tunnels:
                edge_utilization += routed[t.id]
            if edge_utilization.item() > edge.capacity:
                traffic_loss += edge_utilization - torch.tensor(edge.capacity, device=device).float()
        
        if traffic_loss.item() > demand_loss:
            demand_loss = traffic_loss

    return demand_loss

def validate_unavailability(network, solution, real_tm):
    scenarios = network.scenarios
    device = solution.device
    unavailability = torch.tensor(0.0, device=device)
    for s in scenarios:
        # compute amount of traffic on each tunnel
        routed = torch.zeros_like(solution)
        for demand in network.demands.values():
            active_tunnels = [t for t in demand.tunnels if t.pathstr not in s.failed_tunnels]
            if len(active_tunnels) == 0:
                continue
            active_idx = torch.tensor([t.id for t in active_tunnels], device=device, dtype=torch.long)
            ratio_sum = torch.sum(solution[active_idx])
            if ratio_sum.item() == 0:
                num_paths = len(active_tunnels)
                routed[active_idx] = (real_tm[demand.id] * torch.tensor(network.scale, device=device) / num_paths).float()
            else:
                ratio = real_tm[demand.id] * torch.tensor(network.scale, device=device) / ratio_sum
                routed[active_idx] = solution[active_idx] * ratio

        traffic_loss = torch.tensor(0.0, device=device)
        # compute congestion loss
        for edge in network.edges.values():
            edge_utilization = torch.tensor(0.0, device=device)
            for t in edge.tunnels:
                edge_utilization += routed[t.id]
            if edge_utilization.item() > edge.capacity:
                traffic_loss += edge_utilization - torch.tensor(edge.capacity, device=device).float()
        
        if traffic_loss > 1e-3:
            unavailability += traffic_loss * torch.tensor(s.prob, device=device) / traffic_loss.item()

    return unavailability