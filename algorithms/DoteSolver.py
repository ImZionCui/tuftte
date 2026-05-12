import os
import shutil
import subprocess
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from glob import glob

class DoteSolver:
    def __init__(self, network, hist_len=12, function="MAXUTIL"):
        self.network = network
        self.hist_len = hist_len
        self.function = function
        self._num_nodes = len(network.nodes)

    def _copy_data(self):
        topology_name = self.network.name
        if not os.path.exists(f"data/{topology_name}"):
            # Copy the relevant directory to 'DOTE-main' folder
            shutil.copytree(f"../../../data/{topology_name}", f"data/{topology_name}")
            
        if not os.path.exists(f"data/{topology_name}/tunnels.txt"):
            # Write tunnels of current network into file for DOTE to read
            print("[+] Writing tunnels into file.")
            with open(f"data/{topology_name}/tunnels.txt", 'w') as f:
                for src in tqdm(range(self._num_nodes)):
                    for dst in tqdm(range(self._num_nodes)):
                        if src == dst:
                            continue
                        all_paths = self.network.demands[(str(src), str(dst))].tunnels
                        f.write('%d %d:' % (src, dst))
                        tunnel_string = ','.join([t.pathstr for t in all_paths])
                        f.write(tunnel_string + '\n')
                    
    def _compute_opts_to_train(self):
        name = self.network.name
        # compute or aggregate optimums for DOTE to train
        self._copy_data()
        os.chdir(f"data/{name}")
        test_opts_dir = "opts_test"
        train_opts_dir = "opts_train"

        if not os.path.exists(test_opts_dir) or not os.path.exists(train_opts_dir):
            if not os.path.exists(test_opts_dir):
                os.mkdir(test_opts_dir)
            if not os.path.exists(train_opts_dir):
                os.mkdir(train_opts_dir)
            subprocess.run(['python', '../../ml/sl_algos/evaluate.py', '--ecmp_topo', name, '--hist_len', '0', 
                            '--sl_type', 'stats_comm', '--compute_opts'], check=True)
            subprocess.run(['python', '../../ml/sl_algos/evaluate.py', '--ecmp_topo', name, '--hist_len', '0', 
                            '--sl_type', 'eval', '--compute_opts', '--opts_dir', test_opts_dir], check=True)

            subprocess.run(['python', '../../ml/sl_algos/evaluate.py', '--ecmp_topo', name, '--hist_len', '0', 
                            '--sl_type', 'stats_comm', '--compute_opts', '--compute_opts_dir', 'train'], check=True)
            subprocess.run(['python', '../../ml/sl_algos/evaluate.py', '--ecmp_topo', name, '--hist_len', '0', 
                            '--sl_type', 'eval', '--compute_opts', '--compute_opts_dir', 'train', '--opts_dir', train_opts_dir], check=True)

        # Always aggregate numbered opts_* files into per-hist .opt files if missing
        for d in ['test', 'train']:
            opts_info = []
            for file in sorted(glob(d + "/*.hist")):
                with open(file) as f:
                    opts_info.append((file[:-4]+'opt', len(f.readlines())))

            # Skip aggregation when target .opt already exists for all histories
            all_exist = all([os.path.exists(info[0]) for info in opts_info])
            if all_exist:
                continue

            input_file_idx = 0
            for i in range(len(opts_info)):
                opt_res_for_actual_demands = []
                for _ in range(opts_info[i][1]):
                    input_file_name = str(input_file_idx) + '.opt'
                    with(open('opts_' + d + '/' + input_file_name)) as f:
                        lines = f.read().splitlines()
                        for line in lines:
                            if ' Optimal result for actual demand: ' in line:
                                opt_res = float(line[line.find('demand: ')+7:])
                                opt_res_for_actual_demands.append(opt_res)

                    input_file_idx += 1

                with open (opts_info[i][0], 'w') as f:
                    for o in opt_res_for_actual_demands:
                        f.write(str(o) + '\n')

        os.chdir("../..")

    def _train(self):
        name = self.network.name
        os.chdir("algorithms/DOTE-main/networking_envs")
        if not os.path.exists(f"data/{name}/model_dote.pkl"):
            self._compute_opts_to_train()
            # Call 'dote.py' to train
            subprocess.run(['python', 'dote.py', '--ecmp_topo', name, '--opt_function', 
                        self.function, '--hist_len', str(self.hist_len)], check=True)
        # load the model - import NeuralNetworkMaxUtil to ensure it's available during unpickling
        from algorithms.DoteSolver import NeuralNetworkMaxUtil
        model = torch.load(f"data/{name}/model_dote.pkl", weights_only=False)
        model.eval()
        os.chdir("../../..")
        return model

    def solve(self):
        model = self._train()
        # create the dataset
        test_dataset = DmDataset(len(self.network.nodes), self.hist_len, self.network.test_hists._tms)
        # create a data loader for the test set
        test_dl = DataLoader(test_dataset, batch_size=1, shuffle=False)
        with torch.no_grad():
            with tqdm(test_dl) as tests:
                for inputs in tests:
                    pred = model(inputs)
                    self.network.add_sol(pred[0])

# dataset definition
class DmDataset(Dataset):
    def __init__(self, num_nodes, hist_len, tms):
        # store the inputs and outputs
        tms = [np.asarray(tm) for tm in tms]
        np_tms = np.vstack(tms)
        np_tms = np_tms.T
        np_tms_flat = np_tms.flatten('F')

        X_ = []
        for histid in range(len(tms) - hist_len):
            start_idx = histid * num_nodes * (num_nodes - 1)
            end_idx = start_idx + hist_len * num_nodes * (num_nodes - 1)
            X_.append(np_tms_flat[start_idx:end_idx])

        self.X = np.asarray(X_)

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return self.X[idx]

# model definition
class NeuralNetworkMaxUtil(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NeuralNetworkMaxUtil, self).__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.net(x)
        return logits

# def loss_fn_maxutil(y_pred_batch, y_true_batch, env):
#     num_nodes = env.get_num_nodes()
    
#     losses = []
#     loss_vals = []
#     batch_size = y_pred_batch.shape[0]
    
#     for i in range(batch_size):
#         y_pred = y_pred_batch[[i]]
#         y_true = y_true_batch[[i]]
#         opt = y_true[0][num_nodes * (num_nodes - 1)].item()
#         y_true = torch.narrow(y_true, 1, 0, num_nodes * (num_nodes - 1))

    
#         y_pred = y_pred + 1e-16 #eps
#         paths_weight = torch.transpose(y_pred, 0, 1)
#         commodity_total_weight = commodities_to_paths.matmul(paths_weight)
#         commodity_total_weight = 1.0 / (commodity_total_weight)
#         paths_over_total = commodities_to_paths.transpose(0,1).matmul(commodity_total_weight)
#         paths_split = paths_weight.mul(paths_over_total)
#         tmp_demand_on_paths = commodities_to_paths.transpose(0,1).matmul(y_true.transpose(0,1))
#         demand_on_paths = tmp_demand_on_paths.mul(paths_split)
#         flow_on_edges = paths_to_edges.transpose(0,1).matmul(demand_on_paths)
#         congestion = flow_on_edges.divide(torch.tensor(np.array([env._capacities])).transpose(0,1))
#         max_cong = torch.max(congestion)
        
#         loss = 1.0 - max_cong if max_cong.item() == 0.0 else max_cong/max_cong.item()
#         loss_val = 1.0 if opt == 0.0 else max_cong.item() / opt
#         losses.append(loss)
#         loss_vals.append(loss_val)
    
#     ret = sum(losses) / len(losses)
#     ret_val = sum(loss_vals) / len(loss_vals)
    
#     return ret, ret_val