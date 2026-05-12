"""
TUFTTE parameter-scaling solver (based on TUFTTEPredictSolver structure).
Learns theta parameters to scale demand instead of predicting demand directly.
"""

from utils.GurobiSolver import GurobiSolver
from utils.riskMetric import validate_demand_loss, validate_unavailability
from .TEAVARSolver import TEAVARSolver
from .TUFTTESolver_old import Dsolver

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
PRETRAIN_VARIANCE_WEIGHT = 0.05
GRAD_CLIP_NORM = 1.0
BASE_DEMAND_VAR_WEIGHT = 0.3  # how much std influences demand baseline
THETA_CENTER_REG = 0.1  # pull theta toward 1.0 to avoid saturation
DISABLE_THETA_SCALING = False  # set True to bypass learned theta (theta=1)


class TUFTTEParameterSolver:
    """TUFTTE Parameter Learning: learn theta scaling factors, then optimize with CVXPY layer."""
    def __init__(self, network, hist_len=12, type=Availability, suffix=""):
        self.network = network
        self.hist_len = hist_len
        self.name = network.name
        self.type = type
        self.suffix = suffix

    def _history_baseline_from_batch(self, x_batch):
        """从历史向量x构造每个OD的基准需求（不使用真实y）。"""
        x_np = np.asarray(x_batch[0], dtype=np.float32)
        num_pairs = len(x_np) // self.hist_len
        baseline = np.array([
            np.mean(x_np[self.hist_len * i: self.hist_len * (i + 1)])
            for i in range(num_pairs)
        ], dtype=np.float32)
        return baseline

    def _history_baseline_with_var_from_batch(self, x_batch):
        """历史均值 + 方差校正的基准需求，用于容量缩放版本。"""
        x_np = np.asarray(x_batch[0], dtype=np.float32)
        num_pairs = len(x_np) // self.hist_len
        means = []
        stds = []
        for i in range(num_pairs):
            window = x_np[self.hist_len * i: self.hist_len * (i + 1)]
            means.append(np.mean(window))
            stds.append(np.std(window))
        means = np.asarray(means, dtype=np.float32)
        stds = np.asarray(stds, dtype=np.float32)
        return means * (1 + BASE_DEMAND_VAR_WEIGHT * stds / (means + 1e-6))

    def _build_parameter_net(self):
        num_input_pairs = len(self.network.train_hists._tms[0])
        if self.type == Availability:
            output_dim = len(self.network.edges)  # per-edge capacity scaling
            temp_scale = 1.0  # no extra temperature for availability
        else:
            output_dim = len(self.network.edges)  # demand-loss now also scales per-edge capacity
            temp_scale = 20.0  # keep temperature for demand-loss
        return ParameterNN(num_input_pairs, output_dim, self.hist_len, temp_scale=temp_scale)

    def _compute_opts_to_train(self):
        opts_dir = f"data/{self.name}/opts_{self.type}{self.suffix}"
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

    def _pre_train_parameter(self):
        """Pre-train parameter network to output theta close to 1.0 (neutral scaling)."""
        model_name = f"data/{self.name}/model_parameter_scale{self.network.scale}{self.suffix}.pkl"
        if not os.path.exists(model_name):
            print("Training parameter configuration model...")
            data = TUFTTEDataset(self.network.train_hists._tms, [0]*(len(self.network.train_hists._tms)-self.hist_len), self.hist_len)
            train_examples = DataLoader(data, shuffle=True)
            model = self._build_parameter_net()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
            # Availability: neutral; DemandLoss (edge-scaling) also neutral to avoid dim mismatch with OD variance
            if self.type == Availability:
                target_theta_np = np.ones(model.num_output, dtype=np.float32)  # neutral capacity scaling
            else:
                target_theta_np = np.ones(model.num_output, dtype=np.float32)  # per-edge target, no OD variance
            target_theta = torch.tensor(target_theta_np, dtype=torch.float32)
            # Pre-train parameter network to output reasonable initial values
            for ep in range(NUM_EPOCHS):
                with tqdm(train_examples) as tdata:
                    loss_sum = 0
                    count = 0
                    for (x, y, _) in tdata:
                        tdata.set_description(f"Epoch {ep} (Param Pre-train)")
                        optimizer.zero_grad()
                        theta = model(x)
                        # 原始预训练（先注释保留）：鼓励theta接近1.0（中性缩放）
                        # loss = torch.mean((theta - 1.0) ** 2)

                        # 新预训练（混合目标）：
                        # 1) 仍以theta≈1.0为主，防止过度偏移
                        # 2) 以小权重吸收方差信息
                        target_theta_batch = target_theta.to(theta.device).view(1, -1).expand(theta.size(0), -1)
                        neutral_loss = torch.mean((theta - 1.0) ** 2)
                        variance_loss = torch.mean((theta - target_theta_batch) ** 2)
                        loss = neutral_loss + PRETRAIN_VARIANCE_WEIGHT * variance_loss
                        loss.backward()
                        optimizer.step()
                        loss_sum += loss.item()
                        count += 1
                        tdata.set_postfix(loss=loss_sum/count)

            torch.save(model, model_name)

        model = torch.load(model_name)
        return model

    def _train(self):
        opts = self._compute_opts_to_train()
        model_name = f"data/{self.name}/model_{self.type}_parameter_scale{self.network.scale}{self.suffix}.pkl"
        if not os.path.exists(model_name):
            parameter_model = self._pre_train_parameter()
            print(f"Training {self.type} model (parameter-scaling)...")
            data = TUFTTEDataset(self.network.train_hists._tms, opts, self.hist_len)
            train_examples = DataLoader(data, shuffle=True)
            if self.type == DemandLoss:
                model = DemandLossParameterModel(parameter_model, self.network)
            elif self.type == Availability:
                model = TEAVARParameterModel(parameter_model, self.network)
            else:
                raise NotImplementedError
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
            for ep in range(1):
                with tqdm(train_examples) as tdata:
                    loss_sum = 0
                    count = 0
                    for (x, y, opt) in tdata:
                        tdata.set_description(f"Epoch {ep}")
                        optimizer.zero_grad()
                        # 仅使用历史需求x，不使用真实y作为缩放基准
                        x_star, theta = model(x)
                        x_star_used = x_star[0]  # squeeze batch dim

                        if self.type == DemandLoss:
                            loss = validate_demand_loss(self.network, x_star_used, y[0]) - opt.item()
                        elif self.type == Availability:
                            loss = validate_unavailability(self.network, x_star_used, y[0]) - opt.item()

                        if opt.item() > 0.0:
                            loss /= opt.item()
                        # center regularization to keep theta near 1.0
                        center_reg = ((theta - 1.0) ** 2).mean()
                        loss = loss + THETA_CENTER_REG * center_reg
                        # 若loss可导则反向传播；否则跳过以避免RuntimeError
                        if loss.requires_grad:
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                        optimizer.step()
                        loss_sum += loss.item()
                        count += 1
                        tdata.set_postfix(loss=loss_sum/count)

            if self.type in (Availability, DemandLoss):
                # Avoid pickling cvxpylayer by saving only the parameter network weights
                torch.save(model.parameter_model.state_dict(), model_name)
            else:
                torch.save(model, model_name)

        if self.type == Availability:
            parameter_model = self._build_parameter_net()
            state_dict = torch.load(model_name, map_location="cpu")
            parameter_model.load_state_dict(state_dict)
            model = TEAVARParameterModel(parameter_model, self.network)
        elif self.type == DemandLoss:
            parameter_model = self._build_parameter_net()
            state_dict = torch.load(model_name, map_location="cpu")
            parameter_model.load_state_dict(state_dict)
            model = DemandLossParameterModel(parameter_model, self.network)
        else:
            model = torch.load(model_name)
        return model

    def fake_train(self):
        opts = self._compute_opts_to_train()
        parameter_model = self._pre_train_parameter()
        print(f"Trying to train {self.type} model...")
        data = TUFTTEDataset(self.network.train_hists._tms, opts, self.hist_len)
        train_examples = DataLoader(data, shuffle=True)
        if self.type == DemandLoss:
            model = DemandLossParameterModel(parameter_model, self.network)
        elif self.type == Availability:
            model = TEAVARParameterModel(parameter_model, self.network)
        else:
            raise NotImplementedError
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        positive = []
        negative = []
        # get one example
        for (x, y, opt) in train_examples:
            break
        ep = tqdm(range(1000))
        for _ in ep:
            optimizer.zero_grad()
            x_star, theta = model(x)
            x_star_used = x_star[0]
            # Compute bias metrics only when shapes align (skip for edge-scaling)
            pos = 0
            neg = 0
            base_demand = torch.tensor(self._history_baseline_from_batch(x), device=theta.device)
            if base_demand.shape[0] == theta.shape[1]:
                scaled_demand = base_demand * theta[0]
                for i, d in enumerate(scaled_demand):
                    bias = d.item() - y[0][i].item()
                    if bias > 0:
                        pos += bias
                    else:
                        neg += bias

                positive.append(pos)
                negative.append(neg)
            if self.type == DemandLoss:
                loss = validate_demand_loss(self.network, x_star_used, y[0]) - opt.item()
            elif self.type == Availability:
                loss = validate_unavailability(self.network, x_star_used, y[0]) - opt.item()
            if opt.item() > 0.0:
                loss /= opt.item()
            center_reg = ((theta - 1.0) ** 2).mean()
            loss = loss + THETA_CENTER_REG * center_reg
            loss.backward()
            optimizer.step()
            ep.set_postfix(loss=loss.item(), pos=pos, neg=neg)
            if loss.item() == 0.0:
                print(loss)

        return positive, negative

    def output_prediction(self):
        """Output theta parameters (not demand predictions)."""
        model = self._train()
        model.eval()
        fake_opts = [0 for _ in range(len(self.network.test_hists._tms) - self.hist_len)]
        data = TUFTTEDataset(self.network.test_hists._tms, fake_opts, self.hist_len)
        test_instance = DataLoader(data, shuffle=False)
        prediction = []
        with torch.no_grad():
            with tqdm(test_instance) as tdata:
                for (x, _, _) in tdata:
                    theta = model.predict_only(x)
                    prediction.append(theta)

        return prediction

    def solve(self):
        model = self._train()
        model.eval()
        fake_opts = [0 for _ in range(len(self.network.test_hists._tms) - self.hist_len)]
        data = TUFTTEDataset(self.network.test_hists._tms, fake_opts, self.hist_len)
        test_instance = DataLoader(data, shuffle=False, batch_size=1)
        with torch.no_grad():
            with tqdm(test_instance) as tdata:
                sample_idx = 0
                for (x, y, _) in tdata:
                    if self.type == Availability:
                        # Predict per-edge capacity scaling; demand uses history baseline with variance
                        theta = model.predict_only(x)
                        theta_np = np.asarray(theta[0])
                        print(f"\n[Sample {sample_idx}] Edge-capacity thetas ({len(theta_np)} edges):")
                        print(f"  Statistics: Mean={np.mean(theta_np):.4f}, Min={np.min(theta_np):.4f}, Max={np.max(theta_np):.4f}")
                        print(f"  All theta values: {theta_np}")

                        base_demand = self._history_baseline_with_var_from_batch(x)
                        self.network.set_demand_amount(base_demand)

                        # Temporarily scale edge capacities (use fixed edge_list ordering)
                        original_caps = {}
                        for idx, edge in enumerate(model.edge_list):
                            original_caps[idx] = edge.capacity
                            edge.capacity = edge.capacity * theta_np[idx]

                        lp = GurobiSolver()
                        solver = TEAVARSolver(lp, self.network)
                        solver.solve()
                        solution = [t.v_flow_value for t in self.network.solutions.tunnels]
                        self.network.add_sol(solution)

                        # Restore capacities
                        for idx, edge in enumerate(model.edge_list):
                            edge.capacity = original_caps[idx]
                    else:
                        x_star, theta = model(x)
                        theta_np = np.asarray(theta[0])
                        # 打印完整的theta值（现为每条边的容量缩放）
                        print(f"\n[Sample {sample_idx}] Theta values for all {len(theta_np)} edges:")
                        print(f"  Statistics: Mean={np.mean(theta_np):.4f}, Min={np.min(theta_np):.4f}, Max={np.max(theta_np):.4f}")
                        print(f"  All theta values: {theta_np}")

                        # x_star has batch dim from cvxpy layer; squeeze before saving solution
                        x_star_np = x_star[0].detach().cpu().numpy()
                        self.network.add_sol(x_star_np)
                    sample_idx += 1


class TEAVARParameterModel(nn.Module):
    """Parameter-based TEAVAR: learn per-edge capacity scaling factors (demands fixed from history)."""
    def __init__(self, parameter_net, network, beta=0.999):
        super(TEAVARParameterModel, self).__init__()
        self.parameter_model = parameter_net  # Outputs per-edge capacity scale
        num_demands = len(network.demands)
        self.num_demands = num_demands
        num_tunnels = len(network.tunnels)
        num_scenarios = len(network.scenarios)
        self.edge_list = list(network.edges.values())
        self.num_edges = len(self.edge_list)
        self.base_edge_capacity = torch.tensor(
            [edge.capacity for edge in self.edge_list], dtype=torch.float32
        )
        # Parameters to keep DPP: demand matrix (fixed from history) and edge capacity scales
        tm_param = cp.Parameter((1, num_demands), nonneg=True)
        cap_param = cp.Parameter(self.num_edges, nonneg=True)
        x = cp.Variable(num_tunnels, nonneg=True)
        u = cp.Variable(num_scenarios, nonneg=True)
        alpha = cp.Variable(1, nonneg=True)
        problem = self.construct_lp(tm_param, cap_param, x, u, alpha, beta, network)
        assert problem.is_dpp()
        self.cvxlayer = CvxpyLayer(problem, parameters=[tm_param, cap_param], variables=[x, u, alpha])

    def construct_lp(self, tm_param, cap_param, x, u, alpha, beta, network):
        cons = []
        # demand constraints (no demand scaling)
        for i, s in enumerate(network.scenarios):
            for d in network.demands.values():
                flow_on_tunnels = cp.sum([x[t.id] for t in d.tunnels if t.pathstr not in s.failed_tunnels])
                cons.append(flow_on_tunnels >= (1 - alpha[0] - u[i]) * tm_param[0][d.id] * network.scale)
        # edge capacity constraints with learnable scaling
        for idx, edge in enumerate(self.edge_list):
            cons.append(cap_param[idx] >= cp.sum([x[t.id] for t in edge.tunnels]))

        obj = cp.sum([u[i] * s.prob for i, s in enumerate(network.scenarios)]) / (1-beta) + alpha[0]
        problem = cp.Problem(cp.Minimize(obj), cons)
        return problem

    def _history_baseline(self, hist_tms):
        batch_size = hist_tms.shape[0]
        hist = hist_tms.reshape(batch_size, self.num_demands, self.parameter_model.hist_len)
        mean = torch.mean(hist, dim=2)
        std = torch.std(hist, dim=2)
        # incorporate variance to raise baseline cautiously
        return mean * (1 + BASE_DEMAND_VAR_WEIGHT * std / (mean + 1e-6))

    def forward(self, hist_tms):
        # Predict per-edge capacity scaling
        edge_theta = self.parameter_model(hist_tms)
        base_demand = self._history_baseline(hist_tms)
        cap_scaled = self.base_edge_capacity.to(edge_theta.device) * edge_theta
        x_star, _, _ = self.cvxlayer(base_demand, cap_scaled)
        return x_star, edge_theta

    def predict_only(self, hist_tms):
        return self.parameter_model(hist_tms)


class DemandLossParameterModel(nn.Module):
    """Parameter-based Demand Loss: learn per-edge capacity scaling factors."""
    def __init__(self, parameter_net, network):
        super(DemandLossParameterModel, self).__init__()
        self.parameter_model = parameter_net  # Outputs per-edge capacity scale
        num_demands = len(network.demands)
        self.num_demands = num_demands
        num_tunnels = len(network.tunnels)
        num_scenarios = len(network.scenarios)
        self.edge_list = list(network.edges.values())
        self.num_edges = len(self.edge_list)
        self.base_edge_capacity = torch.tensor(
            [edge.capacity for edge in self.edge_list], dtype=torch.float32
        )
        tm_param = cp.Parameter((1, num_demands), nonneg=True)  # demand (fixed from history)
        cap_param = cp.Parameter(self.num_edges, nonneg=True)   # learnable capacity scaling
        x = cp.Variable(num_tunnels, nonneg=True)
        l = cp.Variable((num_scenarios, num_demands), nonneg=True)
        L = cp.Variable(1, nonneg=True)
        problem = self.construct_lp(tm_param, cap_param, x, l, L, network)
        assert problem.is_dpp()
        self.cvxlayer = CvxpyLayer(problem, parameters=[tm_param, cap_param], variables=[x, l, L])

    def construct_lp(self, tm_param, cap_param, x, l, L, network):
        cons = []
        # demand constraints (no demand scaling)
        for i, s in enumerate(network.scenarios):
            for d in network.demands.values():
                flow_on_tunnels = cp.sum([x[t.id] for t in d.tunnels if t.pathstr not in s.failed_tunnels])
                cons.append(flow_on_tunnels >= tm_param[0][d.id] * network.scale - l[i][d.id])
        # edge capacity constraints with learnable scaling
        for idx, edge in enumerate(self.edge_list):
            cons.append(cap_param[idx] >= cp.sum([x[t.id] for t in edge.tunnels]))
        # loss constraints
        cons.append(cp.max(cp.sum(l, axis=1)) <= L)
        problem = cp.Problem(cp.Minimize(L), cons)
        return problem

    def _history_baseline(self, hist_tms):
        batch_size = hist_tms.shape[0]
        hist = hist_tms.reshape(batch_size, self.num_demands, self.parameter_model.hist_len)
        return torch.mean(hist, dim=2)

    def forward(self, hist_tms):
        # Predict per-edge capacity scaling
        edge_theta = self.parameter_model(hist_tms)
        base_demand = self._history_baseline(hist_tms)
        cap_scaled = self.base_edge_capacity.to(edge_theta.device) * edge_theta
        x_star, _, _ = self.cvxlayer(base_demand, cap_scaled)
        return x_star, edge_theta

    def predict_only(self, hist_tms):
        return self.parameter_model(hist_tms)


class ParameterNN(nn.Module):
    """Neural network that outputs control parameters (theta) instead of demand prediction.
    Input: flattened history of OD demands (num_input_pairs * hist_len)
    Output: per-element scaling (dimension = num_output)
    """
    def __init__(self, num_input_pairs, num_output, hist_len=12, temp_scale=20.0):
        super(ParameterNN, self).__init__()
        self.num_input_pairs = num_input_pairs
        self.num_output = num_output
        self.hist_len = hist_len
        self.temp_scale = temp_scale
        self.debug = False  # set True to log raw outputs
        input_dim = num_input_pairs * 2  # mean + std per pair
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_output)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        hist = x.view(batch_size, self.num_input_pairs, self.hist_len)
        mean = torch.mean(hist, dim=2)
        std = torch.std(hist, dim=2)
        feats = torch.cat([mean, std], dim=1)
        raw = self.net(feats)
        if DISABLE_THETA_SCALING:
            return raw * 0.0 + 1.0  # keep grad path but force theta=1
        if self.debug:
            raw_detached = raw.detach()
            flat = raw_detached.view(-1)
            print(
                "[ParameterNN] raw stats: "
                f"mean={raw_detached.mean().item():.4f}, "
                f"min={raw_detached.min().item():.4f}, max={raw_detached.max().item():.4f}, "
                f"sample={flat[:5].cpu().numpy()}"
            )
        raw = raw / self.temp_scale  # temperature to reduce saturation
        softsign = raw / (1 + torch.abs(raw))  # softer than tanh, delays saturation
        theta = 1.0 + 0.1 * softsign  # direct mapping targeting ~[0.9, 1.1]
        return theta


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
