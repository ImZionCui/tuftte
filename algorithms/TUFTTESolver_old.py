"""
Our main design of TUFTTE is in this file.
"""

from utils.GurobiSolver import GurobiSolver
from utils.riskMetric import validate_demand_loss, validate_unavailability
from .TESolver import TESolver
from .TEAVARSolver import TEAVARSolver

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import numpy as np
from tqdm import tqdm

DemandLoss = "D"
Availability = "A"
NUM_EPOCHS = 20

class TUFTTESolver:
    def __init__(self, network, hist_len=12, type=DemandLoss, suffix=""):
        self.network = network
        self.hist_len = hist_len
        self.name = network.name
        self.type = type
        self.suffix = suffix
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # print(f"Using device: {self.device}")

    def _compute_opts_to_train(self):
        opts_dir = f"data/{self.name}/opts_{self.type}"
        if not os.path.exists(opts_dir):
            os.mkdir(opts_dir)

        filename = opts_dir + '/' + self.network.tunnel_type + '_' + str(self.network.scale) + ".opt"
        if not os.path.exists(filename) or os.path.getsize(filename) == 0:
            print("File not found. Computing optimal value for training...")
            with open(filename, 'w') as f:
                if self.type == DemandLoss:
                    for tm in self.network.train_hists._tms[self.hist_len:]:
                        self.network.set_demand_amount(tm)
                        solver = Dsolver(self.network, tm)
                        solver.solve()
                        solution = [t.v_flow_value for t in self.network.solutions.tunnels]
                        loss = validate_demand_loss(self.network, torch.tensor(solution), torch.tensor(tm)).item()
                        f.write(str(loss) + '\n')
                elif self.type == Availability:
                    for tm in self.network.train_hists._tms[self.hist_len:]:
                        self.network.set_demand_amount(tm)
                        lp = GurobiSolver()
                        solver = TEAVARSolver(lp, self.network)
                        solver.solve()
                        solution = [t.v_flow_value for t in self.network.solutions.tunnels]
                        loss = validate_unavailability(self.network, torch.tensor(solution), torch.tensor(tm)).item()
                        f.write(str(loss) + '\n')

        f = open(filename)
        opts = f.read().splitlines()
        f.close()
        opts = [float(opt) for opt in opts]
        return opts[-len(self.network.train_hists)+self.hist_len:]
                
    #训练 ParameterNN 输出合理的初始 theta 值，使 theta 接近 1.0（中性缩放），避免极端的缩放
    def _pre_train(self):
        model_name = f"data/{self.name}/model_parameter_scale{self.network.scale}{self.suffix}.pkl"
        # train a neural network to output control parameters instead of demand prediction
        if not os.path.exists(model_name):
            print("Training parameter configuration model...")
            data = TUFTTEDataset(self.network.train_hists._tms, [0]*(len(self.network.train_hists._tms)-self.hist_len), self.hist_len)
            train_examples = DataLoader(data, shuffle=True)
            l = len(self.network.train_hists._tms[0])
            # model = ParameterNN(l, self.hist_len).to(self.device)
            model = ParameterNN(l, self.hist_len)
            optimizer = torch.optim.Adam(model.parameters())
            # Pre-train parameter network to output reasonable initial values
            for ep in range(NUM_EPOCHS):
                with tqdm(train_examples) as tdata:
                    loss_sum = 0
                    count = 0
                    for (x, y, _) in tdata:
                        tdata.set_description(f"Epoch {ep} (Param Pre-train)")
                        # x, y = x.to(self.device), y.to(self.device)
                        optimizer.zero_grad()
                        theta = model(x)
                        # Encourage theta close to 1.0 initially (neutral scaling)
                        loss = torch.mean((theta - 1.0) ** 2)
                        loss.backward()
                        optimizer.step()
                        loss_sum += loss.item()
                        count += 1
                        tdata.set_postfix(loss=loss_sum/count)

            torch.save(model, model_name)

        model = torch.load(model_name, weights_only=False)
        # model = model.to(self.device)
        return model


    def _train(self):
        opts = self._compute_opts_to_train()
        model_name = f"data/{self.name}/model_{self.type}_scale{self.network.scale}{self.suffix}.pkl"
        if not os.path.exists(model_name):
            predict_model = self._pre_train()
            print(f"Training {self.type} model...")
            data = TUFTTEDataset(self.network.train_hists._tms, opts, self.hist_len)
            train_examples = DataLoader(data, shuffle=True)

            if self.type == DemandLoss:
                # model = DemandLossModel(predict_model, self.network).to(self.device)
                model = DemandLossModel(predict_model, self.network)
            elif self.type == Availability:
                # model = TEAVARModel(predict_model, self.network).to(self.device)
                model = TEAVARModel(predict_model, self.network)
            else:
                raise NotImplementedError
            
            optimizer = torch.optim.Adam(model.parameters())
            for ep in range(1):
                with tqdm(train_examples) as tdata:
                    loss_sum = 0
                    count = 0
                    for (x, y, opt) in tdata:
                        tdata.set_description(f"Epoch {ep}")
                        # x, y, opt = x.to(self.device), y.to(self.device), opt.to(self.device)
                        optimizer.zero_grad()
                        # Key change: pass both hist (x) and true demand (y) to model
                        x_star, theta = model(x, y)
                        if self.type == DemandLoss:
                            loss = validate_demand_loss(self.network, x_star, y[0]) - opt.item()
                        elif self.type == Availability:
                            loss = validate_unavailability(self.network, x_star, y[0]) - opt.item()
                        if opt.item() > 0.0:
                            loss /= opt.item()
                        if loss.item() > 0.0:
                            loss.backward()
                        optimizer.step()
                        loss_sum += loss.item()
                        count += 1
                        tdata.set_postfix(loss=loss_sum/count)

            torch.save(model, model_name)

        model = torch.load(model_name, weights_only=False)
        # model = model.to(self.device)
        return model

    def fake_train(self):
        opts = self._compute_opts_to_train()
        predict_model = self._pre_train()
        print(f"Trying to train {self.type} model...")
        data = TUFTTEDataset(self.network.train_hists._tms, opts, self.hist_len)
        train_examples = DataLoader(data, shuffle=True)
        if self.type == DemandLoss:
            # model = DemandLossModel(predict_model, self.network).to(self.device)
            model = DemandLossModel(predict_model, self.network)
        elif self.type == Availability:
            # model = TEAVARModel(predict_model, self.network).to(self.device)
            model = TEAVARModel(predict_model, self.network)
        else:
            raise NotImplementedError
        optimizer = torch.optim.Adam(model.parameters())
        positive = []
        negative = []
        # get one example
        for (x, y, opt) in train_examples:
            break
        # x, y, opt = x.to(self.device), y.to(self.device), opt.to(self.device)
        for _ in ep:
            optimizer.zero_grad()
            x_star, pred = model(x)
            pos = 0
            neg = 0
            for i, d in enumerate(pred[0]):
                bias = d.item() - y[0][i].item()
                if bias > 0:
                    pos += bias
                else:
                    neg += bias
            
            positive.append(pos)
            negative.append(neg)
            if self.type == DemandLoss:
                loss = validate_demand_loss(self.network, x_star, y[0]) - opt.item()
            elif self.type == Availability:
                loss = validate_unavailability(self.network, x_star, y[0]) - opt.item()
            if opt.item() > 0.0:
                loss /= opt.item()
            if loss.item() == 0.0:
                print(loss)
                print(opt)
                break
            loss.backward()
            optimizer.step()
            ep.set_postfix(loss=loss.item(), pos=pos, neg=neg) # Update progress bar with bias info

        return positive, negative
    
    def output_prediction(self):
        model = self._train()
        model.eval()
        fake_opts = [0 for _ in range(len(self.network.test_hists._tms) - self.hist_len)]
        data = TUFTTEDataset(self.network.test_hists._tms, fake_opts, self.hist_len)
        test_instance = DataLoader(data, shuffle=False)
        prediction = []
        with torch.no_grad():
            with tqdm(test_instance) as tdata:
                for (x, _, _) in tdata:
                    # x = x.to(self.device)
                    pred = model.predict_only(x)
                    prediction.append(pred)

        return prediction
    
    def solve(self):
        model = self._train()
        model.eval()
        fake_opts = [0 for _ in range(len(self.network.test_hists._tms) - self.hist_len)]
        data = TUFTTEDataset(self.network.test_hists._tms, fake_opts, self.hist_len)
        test_instance = DataLoader(data, shuffle=False)
        with torch.no_grad():
            with tqdm(test_instance) as tdata:
                for (x, y, _) in tdata:
                    # x = x.to(self.device)
                    if self.type == Availability:
                        pred = model.predict_only(x)
                        # move prediction to CPU before converting to numpy
                        # pred_np = pred.detach().cpu().numpy()
                        self.network.set_demand_amount(pred_np[0])
                        lp = GurobiSolver()
                        solver = TEAVARSolver(lp, self.network)
                        solver.solve()
                        solution = [t.v_flow_value for t in self.network.solutions.tunnels]
                        self.network.add_sol(solution)
                    else:
                        x_star, theta = model(x, y)
                        # move x_star to CPU before converting to numpy
                        x_star_np = x_star.detach().cpu().numpy()
                        self.network.add_sol(x_star_np)


# class TUFTTEPredictSolver:
#     """Original TUFTTE: learn to predict demand, then optimize with CVXPY layer."""
#     def __init__(self, network, hist_len=12, type=Availability, suffix=""):
#         self.network = network
#         self.hist_len = hist_len
#         self.name = network.name
#         self.type = type
#         self.suffix = suffix
#         # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         # print(f"Using device: {self.device}")

#     def _compute_opts_to_train(self):
#         opts_dir = f"data/{self.name}/opts_{self.type}{self.suffix}"
#         if not os.path.exists(opts_dir):
#             os.mkdir(opts_dir)

#         filename = opts_dir + '/' + self.network.tunnel_type + '_' + str(self.network.scale) + ".opt"
#         if not os.path.exists(filename) or os.path.getsize(filename) == 0:
#             print("File not found. Computing optimal value for training...")
#             with open(filename, 'w') as f:
#                 if self.type == DemandLoss:
#                     for tm in self.network.train_hists._tms[self.hist_len:]:
#                         self.network.set_demand_amount(tm)
#                         solver = Dsolver(self.network, tm)
#                         solver.solve()
#                         solution = [t.v_flow_value for t in self.network.solutions.tunnels]
#                         loss = validate_demand_loss(self.network, torch.tensor(solution), torch.tensor(tm)).item()
#                         f.write(str(loss) + '\n')
#                 elif self.type == Availability:
#                     for tm in self.network.train_hists._tms[self.hist_len:]:
#                         self.network.set_demand_amount(tm)
#                         lp = GurobiSolver()
#                         solver = TEAVARSolver(lp, self.network)
#                         solver.solve()
#                         solution = [t.v_flow_value for t in self.network.solutions.tunnels]
#                         loss = validate_unavailability(self.network, torch.tensor(solution), torch.tensor(tm)).item()
#                         f.write(str(loss) + '\n')

#         f = open(filename)
#         opts = f.read().splitlines()
#         f.close()
#         opts = [float(opt) for opt in opts]
#         return opts[-len(self.network.train_hists)+self.hist_len:]

#     def _pre_train_predict(self):
#         model_name = f"data/{self.name}/model_predict{self.suffix}.pkl"
#         if not os.path.exists(model_name):
#             print("Training demand prediction model...")
#             data = PredictDataset(self.network.train_hists._tms, self.hist_len)
#             train_examples = DataLoader(data, shuffle=True)
#             l = len(self.network.train_hists._tms[0])
#             model = DemandPredictNN(l, self.hist_len)
#             optimizer = torch.optim.Adam(model.parameters())
#             for ep in range(NUM_EPOCHS):
#                 with tqdm(train_examples) as tdata:
#                     loss_sum = 0
#                     count = 0
#                     for (x, y) in tdata:
#                         tdata.set_description(f"Epoch {ep} (Demand Predict)")
#                         # x, y = x.to(self.device), y.to(self.device)
#                         optimizer.zero_grad()
#                         y_hat = model(x)
#                         norm = torch.sum(y ** 2) / l
#                         loss = nn.MSELoss()(y, y_hat) / norm.item()
#                         loss.backward()
#                         optimizer.step()
#                         loss_sum += loss.item()
#                         count += 1
#                         tdata.set_postfix(loss=loss_sum/count)

#             torch.save(model, model_name)

#         # model = torch.load(model_name, weights_only=False)
#         model = torch.load(model_name)

#         # model = model.to(self.device)
#         return model

#     def _train(self):
#         opts = self._compute_opts_to_train()
#         model_name = f"data/{self.name}/model_{self.type}_predict{self.suffix}.pkl"
#         if not os.path.exists(model_name):
#             predict_model = self._pre_train_predict()
#             print(f"Training {self.type} model (predict-demand)...")
#             data = TUFTTEDataset(self.network.train_hists._tms, opts, self.hist_len)
#             train_examples = DataLoader(data, shuffle=True)
#             if self.type == DemandLoss:
#                 model = DemandLossModel(predict_model, self.network)
#             elif self.type == Availability:
#                 model = TEAVARModel(predict_model, self.network)
#             else:
#                 raise NotImplementedError
#             optimizer = torch.optim.Adam(model.parameters())
#             for ep in range(1):
#                 with tqdm(train_examples) as tdata:
#                     loss_sum = 0
#                     count = 0
#                     for (x, y, opt) in tdata:
#                         tdata.set_description(f"Epoch {ep}")
#                         # x, y, opt = x.to(self.device), y.to(self.device), opt.to(self.device)
#                         optimizer.zero_grad()
#                         x_star, _ = model(x)

#                         if self.type == DemandLoss:
#                             loss = validate_demand_loss(self.network, x_star, y[0]) - opt.item()
#                         elif self.type == Availability:
#                             loss = validate_unavailability(self.network, x_star, y[0]) - opt.item()
                        
#                         if opt.item() > 0.0:
#                             loss /= opt.item()
#                         if loss.item() > 0.0:
#                             loss.backward()
#                         optimizer.step()
#                         loss_sum += loss.item()
#                         count += 1
#                         tdata.set_postfix(loss=loss_sum/count)

#             torch.save(model, model_name)

#         model = torch.load(model_name)
#         # model = model.to(self.device)
#         return model
    
#     def fake_train(self):
#         opts = self._compute_opts_to_train()
#         predict_model = self._pre_train()
#         print(f"Trying to train {self.type} model...")
#         data = TUFTTEDataset(self.network.train_hists._tms, opts, self.hist_len)
#         train_examples = DataLoader(data, shuffle=True)
#         if self.type == DemandLoss:
#             # DemandLossPredictModel not implemented
#             # raise NotImplementedError("DemandLoss mode not supported in TUFTTEPredictSolver")
#             model = DemandLossModel(predict_model, self.network)
#         elif self.type == Availability:
#             model = TEAVARModel(predict_model, self.network)
#         else:
#             raise NotImplementedError
#         optimizer = torch.optim.Adam(model.parameters())
#         positive = []
#         negative = []
#         # get one example
#         for (x, y, opt) in train_examples:
#             break
#         # x, y, opt = x.to(self.device), y.to(self.device), opt.to(self.device)
#         ep = tqdm(range(1000))
#         for _ in ep:
#             optimizer.zero_grad()
#             x_star, pred = model(x)
#             pos = 0
#             neg = 0
#             for i, d in enumerate(pred[0]):
#                 bias = d.item() - y[0][i].item()
#                 if bias > 0:
#                     pos += bias
#                 else:
#                     neg += bias
            
#             positive.append(pos)
#             negative.append(neg)
#             if self.type == DemandLoss:
#                 loss = validate_demand_loss(self.network, x_star, y[0]) - opt.item()
#             elif self.type == Availability:
#                 loss = validate_unavailability(self.network, x_star, y[0]) - opt.item()
#             if opt.item() > 0.0:
#                 loss /= opt.item()
#             if loss.item() == 0.0:
#                 print(loss)
#                 print(opt)
#                 break
#             loss.backward()
#             optimizer.step()
#             ep.set_postfix(loss=loss.item(), pos=pos, neg=neg)

#         return positive, negative
    
#     def output_prediction(self):
#         model = self._train()
#         model.eval()
#         fake_opts = [0 for _ in range(len(self.network.test_hists._tms) - self.hist_len)]
#         data = TUFTTEDataset(self.network.test_hists._tms, fake_opts, self.hist_len)
#         test_instance = DataLoader(data, shuffle=False)
#         prediction = []
#         with torch.no_grad():
#             with tqdm(test_instance) as tdata:
#                 for (x, _, _) in tdata:
#                     # x = x.to(self.device)
#                     pred = model.predict_only(x)
#                     prediction.append(pred)

#         return prediction

#     def solve(self):
#         model = self._train()
#         model.eval()
#         fake_opts = [0 for _ in range(len(self.network.test_hists._tms) - self.hist_len)]
#         data = TUFTTEDataset(self.network.test_hists._tms, fake_opts, self.hist_len)
#         test_instance = DataLoader(data, shuffle=False)
#         with torch.no_grad():
#             with tqdm(test_instance) as tdata:
#                 for (x, y, _) in tdata:
#                     # x = x.to(self.device)
#                     if self.type == Availability:
#                         pred = model.predict_only(x)
#                         # move prediction to CPU before converting to numpy
#                         # pred_np = pred.detach().cpu().numpy()
#                         self.network.set_demand_amount(np.asarray(pred[0]))
#                         lp = GurobiSolver()
#                         solver = TEAVARSolver(lp, self.network)
#                         solver.solve()
#                         solution = [t.v_flow_value for t in self.network.solutions.tunnels]
#                         self.network.add_sol(solution)
#                     else:
#                         x_star, pred = model(x)  # Ensure x is on the correct device
#                         # move x_star to CPU before converting to numpy
#                         # x_star_np = x_star.detach().cpu().numpy()
#                         # self.network.add_sol(x_star_np)
#                         self.network.add_sol(np.asarray(x_star))

class Dsolver(TESolver):   #求解demand loss模型的优化问题
    def __init__(self, network, tm):
        lp = GurobiSolver()
        TESolver.__init__(self, lp, network)
        self.tm = tm
        self.L = lp.Variable(lb=0)  # overall loss variable
        self.l_s = [lp.Variables(len(network.demands), lb=0) for _ in network.scenarios] # per-scenario, per-demand loss variables

    def add_demand_constraints(self):
        for i, s in enumerate(self.network.scenarios):
            for j, d in enumerate(self.network.demands.values()):
                flow_on_tunnels = sum([t.v_flow for t in d.tunnels if t.pathstr not in s.failed_tunnels])
                self.lp.Assert(flow_on_tunnels >= d.amount * self.network.scale - self.l_s[i][j])
 
    def add_loss_constraints(self):
        for l in self.l_s:
            self.lp.Assert(self.lp.Sum(l) <= self.L)

    def solve(self):
        self.add_demand_constraints()
        self.add_edge_capacity_constraints()
        self.add_loss_constraints()
        self.Minimize(self.L)
        obj = self.lp.Solve()
        self.network.set_tunnel_flow(self.lp.Value)
        return obj
    
class DemandLossModel(nn.Module):  #在demandloss目标下，学习一个控制参数 theta 来调整需求预测，避免过度缩放导致的极端解
    def __init__(self, predict_net, network):
        super(DemandLossModel, self).__init__()
        self.predict_model = predict_net  
        num_demands = len(network.demands)
        num_tunnels = len(network.tunnels)
        num_scenarios = len(network.scenarios)
        tm = cp.Parameter((1, num_demands), nonneg=True)  # True demand
        # theta = cp.Parameter((1, num_demands), nonneg=True)  # Demand scaling parameter
        x = cp.Variable(num_tunnels, nonneg=True)
        l = cp.Variable((num_scenarios, num_demands), nonneg=True)
        L = cp.Variable(1, nonneg=True)
        problem = self.construct_lp(tm, x, l, L, network)
        assert problem.is_dpp()
        self.cvxlayer = CvxpyLayer(problem, parameters=[tm], variables=[x, l, L])

    def construct_lp(self, tm, x, l, L, network):
        cons = []
        # add demand constraints with learnable scaling parameter theta
        for i, s in enumerate(network.scenarios):
            for d in network.demands.values():
                flow_on_tunnels = cp.sum([x[t.id] for t in d.tunnels if t.pathstr not in s.failed_tunnels])
                # Key change: scale demand by theta (learned parameter)
                cons.append(flow_on_tunnels >= tm[0][d.id] * network.scale - l[i][d.id])
        # add edge capacity constraints
        for edge in network.edges.values():
            cons.append(edge.capacity >= cp.sum([x[t.id] for t in edge.tunnels]))
        # add loss constraints
        cons.append(cp.max(cp.sum(l, axis=1)) <= L)
        problem = cp.Problem(cp.Minimize(L), cons)
        return problem

    def forward(self, hist_tms):
        # Predict control parameter theta from history
        prediction = self.predict_model(hist_tms)
        # Solve optimization with true demand and learned parameter
        x_star, _, _ = self.cvxlayer(prediction)
        return x_star, prediction
    
    def predict_only(self, hist_tms):
        return self.predict_model(hist_tms)

class TEAVARModel(nn.Module):
    def __init__(self, parameter_net, network, beta=0.999):
        super(TEAVARModel, self).__init__()
        self.parameter_model = parameter_net  # Outputs control parameters
        num_demands = len(network.demands)
        num_tunnels = len(network.tunnels)
        num_scenarios = len(network.scenarios)
        tm_scaled = cp.Parameter((1, num_demands), nonneg=True)  # Scaled demand (tm * theta)
        x = cp.Variable(num_tunnels, nonneg=True)
        u = cp.Variable(num_scenarios, nonneg=True)
        alpha = cp.Variable(1, nonneg=True)
        problem = self.construct_lp(tm_scaled, x, u, alpha, beta, network)
        assert problem.is_dpp()
        self.cvxlayer = CvxpyLayer(problem, parameters=[tm_scaled], variables=[x, u, alpha])
        
    def construct_lp(self, tm_scaled, x, u, alpha, beta, network):
        cons = []
        # add demand constraints with learnable scaling parameter theta
        for i, s in enumerate(network.scenarios):
            for d in network.demands.values():
                flow_on_tunnels = cp.sum([x[t.id] for t in d.tunnels if t.pathstr not in s.failed_tunnels])
                # Scaled demand provided as a single parameter to satisfy DPP
                cons.append(flow_on_tunnels >= (1 - alpha[0] - u[i]) * tm_scaled[0][d.id] * network.scale)
        # add edge capacity constraints
        for edge in network.edges.values():
            cons.append(edge.capacity >= cp.sum([x[t.id] for t in edge.tunnels]))
        # define objective
        obj = cp.sum([u[i] * s.prob for i, s in enumerate(network.scenarios)]) / (1-beta) + alpha[0]
        problem = cp.Problem(cp.Minimize(obj), cons)
        return problem
    
    def forward(self, hist_tms, true_demand):
        # Predict control parameter theta from history
        theta = self.parameter_model(hist_tms)
        # Solve optimization with scaled demand (tm * theta)
        tm_scaled = true_demand * theta
        x_star, _, _ = self.cvxlayer(tm_scaled)
        return x_star, theta
    
    def predict_only(self, hist_tms):
        return self.parameter_model(hist_tms)

# class TEAVARPredictModel(nn.Module):
#     """Original TUFTTE: predict demand, then optimize with CVXPY layer."""
   
#     def __init__(self, predict_net, network, beta=0.999):
#         super(TEAVARModel, self).__init__()
#         self.predict_model = predict_net
#         num_demands = len(network.demands)
#         num_tunnels = len(network.tunnels)
#         num_scenarios = len(network.scenarios)
#         tm = cp.Parameter((1, num_demands), nonneg=True)
#         x = cp.Variable(num_tunnels, nonneg=True)
#         u = cp.Variable(num_scenarios, nonneg=True)
#         alpha = cp.Variable(1, nonneg=True)
#         problem = self.construct_lp(tm, x, u, alpha, beta, network)
#         assert problem.is_dpp()
#         self.cvxlayer = CvxpyLayer(problem, parameters=[tm], variables=[x, u, alpha])

#     def construct_lp(self, tm_pred, x, u, alpha, beta, network):
#         cons = []
#         for i, s in enumerate(network.scenarios):
#             for d in network.demands.values():
#                 flow_on_tunnels = cp.sum([x[t.id] for t in d.tunnels if t.pathstr not in s.failed_tunnels])
#                 cons.append(flow_on_tunnels >= (1 - alpha - u[i]) * tm_pred[0][d.id] * network.scale)
#         for edge in network.edges.values():
#             cons.append(edge.capacity >= cp.sum([x[t.id] for t in edge.tunnels]))
#         obj = cp.sum([u[i] * s.prob for i, s in enumerate(network.scenarios)]) / (1-beta) + alpha
#         problem = cp.Problem(cp.Minimize(obj), cons)
#         return problem

   
#     def forward(self, hist_tms):
#         prediction = self.predict_model(hist_tms)
#         x_star, _, _ = self.cvxlayer(prediction)
#         return x_star, prediction
    
#     def predict_only(self, hist_tms):
#         return self.predict_model(hist_tms)
    

class ParameterNN(nn.Module):
    """Neural network that outputs control parameters (theta) instead of demand prediction.
    theta: demand scaling factor in range [0.5, 1.5] for each OD pair.
    """
    def __init__(self, num_pairs, hist_len=12):
        super(ParameterNN, self).__init__()
        self.num_pairs = num_pairs
        self.hist_len = hist_len
        self.net = nn.Sequential(
            nn.Linear(num_pairs * hist_len, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_pairs),
            nn.Sigmoid()  # Output in [0, 1]
        )

    def forward(self, x):
        # Output theta in range [0.7, 1.3] - tighter than original [0.5, 1.5]
        normalized = self.net(x)  # [0, 1]
        theta = 0.5 + normalized  # [0.5, 1.5]
        return theta
    
        res = [torch.mean(x[0][self.hist_len * i: self.hist_len * (i+1)]) for i in range(self.num_pairs)]
        # output = self.net(x) + torch.tensor(res, device=x.device)
        output = self.net(x) + torch.tensor(res)
        return output
    
class PredictDataset(Dataset):
    def __init__(self, tms, hist_len=12):
        X_ = []
        for idx in range(len(tms) - hist_len):
            X_.append(np.stack(tms[idx:idx + hist_len]).flatten('F'))
        self.X = np.asarray(X_, dtype=np.float32)
        self.y = np.asarray(tms[hist_len:], dtype=np.float32)

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx])
    
class TUFTTEDataset(Dataset):
    def __init__(self, tms, opts, hist_len=12):
        X_ = []
        for idx in range(len(tms) - hist_len):
            X_.append(np.stack(tms[idx:idx + hist_len]).flatten('F'))
        self.X = np.asarray(X_, dtype=np.float32)
        self.y = np.asarray(tms[hist_len:], dtype=np.float32)
        self.opt = np.asarray(opts)

    def __len__(self):
        return len(self.opt)
    
    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx], self.opt[idx])