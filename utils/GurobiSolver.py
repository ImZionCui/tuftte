from gurobipy import Model, GRB, quicksum

class GurobiSolver:
    def __init__(self):
        self.problem = Model()
        self.problem.Params.OutputFlag = 0
        self.Sum = quicksum

    def Variable(self, lb = -GRB.INFINITY, type = None):
        if type == "Int":
            return self.problem.addVar(lb=lb, vtype=GRB.INTEGER)
        elif type == "Bool":
            return self.problem.addVar(lb=lb, vtype=GRB.BINARY)
        return self.problem.addVar(lb=lb)

    def Variables(self, shape = 1, lb = -GRB.INFINITY, type=None):
        if type == "Int":
            return self.problem.addVars(shape, lb=lb, vtype=GRB.INTEGER)
        elif type == "Bool":
            return self.problem.addVars(shape, lb=lb, vtype=GRB.BINARY)
        return self.problem.addVars(shape, lb=lb)
    
    def Maximize(self, objective):
        self.problem.setObjective(objective, GRB.MAXIMIZE)

    def Minimize(self, objective):
        self.problem.setObjective(objective, GRB.MINIMIZE)

    def Assert(self, constraint):
        self.problem.addConstr(constraint)

    def Solve(self):
        self.problem.optimize()
        if self.problem.status == GRB.Status.OPTIMAL:
            print("Optimal solution was found.")
            return self.problem.ObjVal
        elif self.problem.status == GRB.Status.INFEASIBLE:
            print("Model is infeasible.")
            # 计算 IIS 帮助诊断
            self.problem.computeIIS()
            print("IIS computed, check constraints.")
            raise Exception("Model infeasible")
        elif self.problem.status == GRB.Status.UNBOUNDED:
            print("Model is unbounded.")
            raise Exception("Model unbounded")
        else:
            print(f"Optimal solution was not found. Status: {self.problem.status}")
            raise Exception(f"Solver status: {self.problem.status}")
    
    def SetStart(self, var_dict):
        """设置热启动初始值（变量字典）"""
        for var, value in var_dict.items():
            try:
                var.Start = float(value)
            except:
                pass  # 忽略无效变量
    
    def GetSolution(self):
        """获取当前解的变量值字典，用于下一步热启动"""
        solution = {}
        try:
            for var in self.problem.getVars():
                solution[var] = var.X
        except:
            pass
        return solution
    
    
    
    def Value(self, var):
        return var.x