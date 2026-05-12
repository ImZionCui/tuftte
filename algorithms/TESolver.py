"""
TESolver serves as a basic framework, including three common optimization objectives:
    MaxFlow: Maximize the throughput;
    MaxMin: Minimize the max-min fairness;
    MLU: Minimize the maximum link utilization (MLU).

FFCSolver and TEAVARSolver are implemented based on TESolver.
"""

class TESolver:
    def __init__(self, lp, network):
        self.lp = lp
        self.network = network
        self.prev_solution = None  # 用于热启动的前一步解
        self.initialize_optimization_variables()
    
    def initialize_optimization_variables(self):
        for tunnel in self.network.tunnels.values():
            tunnel.init_flow_var(self.lp)
    
    def add_demand_constraints(self):
        for demand in self.network.demands.values():
            flow_on_tunnels = self.lp.Sum([tunnel.v_flow for tunnel in demand.tunnels])
            assert len(demand.tunnels) > 0
            self.lp.Assert(flow_on_tunnels == demand.amount * self.network.scale)

    def add_edge_capacity_constraints(self):
        for edge in self.network.edges.values():
            # Use linear expression with solver Sum to avoid TempConstr bool issues
            flow_expr = self.lp.Sum([t.v_flow for t in edge.tunnels])
            self.lp.Assert(flow_expr <= edge.capacity)
                    
    def Maximize(self, objective):
        self.lp.Maximize(objective)

    def Minimize(self, objective):
        self.lp.Minimize(objective)
    
    def set_warm_start(self, warm_start_solution):
        """设置热启动初始解"""
        self.prev_solution = warm_start_solution
        
    def solve(self, obj="MaxFlow"):
        # 如果有前一步解，使用热启动
        if self.prev_solution:
            self.lp.SetStart(self.prev_solution)
        
        if obj == "MaxFlow":
            self.add_demand_constraints()
            self.add_edge_capacity_constraints()
            self.Maximize(sum([t.v_flow for t in self.network.tunnels.values()]))
        elif obj == "MaxMin":
            amin = self.lp.Variable()
            self.lp.Assert(amin <= 1)
            self.add_edge_capacity_constraints()
            for demand in self.network.demands.values():
                flow_on_tunnels = sum([tunnel.v_flow for tunnel in demand.tunnels])
                assert len(demand.tunnels) > 0
                self.lp.Assert(amin * demand.amount * self.network.scale == flow_on_tunnels)

            self.Maximize(amin)
        elif obj == "MLU":
            Z = self.lp.Variable()
            self.add_demand_constraints()
            for edge in self.network.edges.values():
                flow_expr = self.lp.Sum([t.v_flow for t in edge.tunnels])
                self.lp.Assert(flow_expr <= Z * edge.capacity)

            self.Minimize(Z)
        else:
            print(f"Objective {obj} is not defined!")
            raise NotImplementedError

        obj = self.lp.Solve()
        
        # 保存当前解作为下一步的热启动初始值
        self.prev_solution = self.lp.GetSolution()
        
        self.network.set_tunnel_flow(self.lp.Value)
        return obj