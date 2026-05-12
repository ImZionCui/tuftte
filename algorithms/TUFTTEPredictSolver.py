"""
TUFTTE predict-demand solver (separated from TUFTTESolver).
"""

from utils.GurobiSolver import GurobiSolver
from utils.riskMetric import validate_demand_loss, validate_unavailability
from .TEAVARSolver import TEAVARSolver
from .TUFTTESolver_old import Dsolver, DemandLossModel

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


class TUFTTEPredictSolver:
    """Original TUFTTE: learn to predict demand, then optimize with CVXPY layer."""
    def __init__(self, network, hist_len=12, type=Availability, suffix=""):
        self.network = network
        self.hist_len = hist_len
        self.name = network.name
        self.type = type
        self.suffix = suffix

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

    def _pre_train_predict(self):
        model_name = f"data/{self.name}/model_predict_scale{self.network.scale}{self.suffix}.pkl"
        if not os.path.exists(model_name):
            print("Training demand prediction model...")
            data = PredictDataset(self.network.train_hists._tms, self.hist_len)
            train_examples = DataLoader(data, shuffle=True)
            l = len(self.network.train_hists._tms[0])
            model = DemandPredictNN(l, self.hist_len)
            optimizer = torch.optim.Adam(model.parameters())
            for ep in range(NUM_EPOCHS):
                with tqdm(train_examples) as tdata:
                    loss_sum = 0
                    count = 0
                    for (x, y) in tdata:
                        tdata.set_description(f"Epoch {ep} (Demand Predict)")
                        optimizer.zero_grad()
                        y_hat = model(x)
                        norm = torch.sum(y ** 2) / l
                        loss = nn.MSELoss()(y, y_hat) / norm.item()
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
        model_name = f"data/{self.name}/model_{self.type}_predict_scale{self.network.scale}{self.suffix}.pkl"
        if not os.path.exists(model_name):
            predict_model = self._pre_train_predict()
            print(f"Training {self.type} model (predict-demand)...")
            data = TUFTTEDataset(self.network.train_hists._tms, opts, self.hist_len)
            train_examples = DataLoader(data, shuffle=True)
            if self.type == DemandLoss:
                model = DemandLossModel(predict_model, self.network)
            elif self.type == Availability:
                model = TEAVARPredictModel(predict_model, self.network)
            else:
                raise NotImplementedError
            optimizer = torch.optim.Adam(model.parameters())
            for ep in range(1):
                with tqdm(train_examples) as tdata:
                    loss_sum = 0
                    count = 0
                    for (x, y, opt) in tdata:
                        tdata.set_description(f"Epoch {ep}")
                        optimizer.zero_grad()
                        x_star, _ = model(x)

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

        model = torch.load(model_name)
        return model

    def fake_train(self):
        opts = self._compute_opts_to_train()
        predict_model = self._pre_train_predict()
        print(f"Trying to train {self.type} model...")
        data = TUFTTEDataset(self.network.train_hists._tms, opts, self.hist_len)
        train_examples = DataLoader(data, shuffle=True)
        if self.type == DemandLoss:
            model = DemandLossModel(predict_model, self.network)
        elif self.type == Availability:
            model = TEAVARPredictModel(predict_model, self.network)
        else:
            raise NotImplementedError
        optimizer = torch.optim.Adam(model.parameters())
        positive = []
        negative = []
        # get one example
        for (x, y, opt) in train_examples:
            break
        ep = tqdm(range(1000))
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
            ep.set_postfix(loss=loss.item(), pos=pos, neg=neg)

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
                    if self.type == Availability:
                        pred = model.predict_only(x)
                        self.network.set_demand_amount(np.asarray(pred[0]))
                        lp = GurobiSolver()
                        solver = TEAVARSolver(lp, self.network)
                        solver.solve()
                        solution = [t.v_flow_value for t in self.network.solutions.tunnels]
                        self.network.add_sol(solution)
                    else:
                        x_star, pred = model(x)
                        self.network.add_sol(np.asarray(x_star))


class TEAVARPredictModel(nn.Module):
    """Original TUFTTE: predict demand, then optimize with CVXPY layer."""
    def __init__(self, predict_net, network, beta=0.999):
        super(TEAVARPredictModel, self).__init__()
        self.predict_model = predict_net
        num_demands = len(network.demands)
        num_tunnels = len(network.tunnels)
        num_scenarios = len(network.scenarios)
        tm = cp.Parameter((1, num_demands), nonneg=True)
        x = cp.Variable(num_tunnels, nonneg=True)
        u = cp.Variable(num_scenarios, nonneg=True)
        alpha = cp.Variable(1, nonneg=True)
        problem = self.construct_lp(tm, x, u, alpha, beta, network)
        assert problem.is_dpp()
        self.cvxlayer = CvxpyLayer(problem, parameters=[tm], variables=[x, u, alpha])

    def construct_lp(self, tm_pred, x, u, alpha, beta, network):
        cons = []
        for i, s in enumerate(network.scenarios):
            for d in network.demands.values():
                flow_on_tunnels = cp.sum([x[t.id] for t in d.tunnels if t.pathstr not in s.failed_tunnels])
                cons.append(flow_on_tunnels >= (1 - alpha - u[i]) * tm_pred[0][d.id] * network.scale)
        for edge in network.edges.values():
            cons.append(edge.capacity >= cp.sum([x[t.id] for t in edge.tunnels]))
        obj = cp.sum([u[i] * s.prob for i, s in enumerate(network.scenarios)]) / (1-beta) + alpha
        problem = cp.Problem(cp.Minimize(obj), cons)
        return problem

    def forward(self, hist_tms):
        prediction = self.predict_model(hist_tms)
        x_star, _, _ = self.cvxlayer(prediction)
        return x_star, prediction

    def predict_only(self, hist_tms):
        return self.predict_model(hist_tms)


class DemandPredictNN(nn.Module):
    """Neural network that predicts demand directly from history."""
    def __init__(self, num_pairs, hist_len=12):
        super(DemandPredictNN, self).__init__()
        self.num_pairs = num_pairs
        self.hist_len = hist_len
        self.net = nn.Sequential(
            nn.Linear(num_pairs * hist_len, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_pairs)
        )

    def forward(self, x):
        res = [torch.mean(x[0][self.hist_len * i: self.hist_len * (i+1)]) for i in range(self.num_pairs)]
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