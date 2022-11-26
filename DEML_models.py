import torch.nn.functional as F
import torch
from torch import nn, optim
from torch.nn.modules.container import ModuleList


class Expert(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Expert, self).__init__()
        self.fc1 = nn.Sequential(nn.Dropout(0.2),nn.Linear(input_size, hidden_size),nn.BatchNorm1d(hidden_size),nn.ReLU(),nn.Dropout(0.5))
        self.fc2 = nn.Sequential(nn.Linear(hidden_size, output_size),nn.BatchNorm1d(output_size),nn.ReLU(),nn.Dropout(0.5))
    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        return out
class Expert_MM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Expert_MM, self).__init__()
        self.layer1=Expert(int(input_size*0.5),output_size,hidden_size)
        self.layer2=Expert(int(input_size*0.5),output_size,hidden_size)
    def forward(self,x):
        length=int(0.5*x.shape[-1])
        x1,x2=x[:,0:length],x[:,length::]
        x1=self.layer1(x1)
        x2=self.layer2(x2)
        return x1+x2
class Expert_BI(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Expert_BI, self).__init__()
        self.layer1=Expert(int(input_size*0.5),output_size,hidden_size)
        self.layer2=Expert(int(input_size*0.5),output_size,hidden_size)
        self.aux1=nn.Sequential(nn.Linear(output_size, output_size),nn.BatchNorm1d(output_size),nn.ReLU(),nn.Dropout(0.5))
        self.aux2=nn.Sequential(nn.Linear(output_size, output_size),nn.BatchNorm1d(output_size),nn.ReLU(),nn.Dropout(0.5))
    def forward(self,x):
        length=int(0.5*x.shape[-1])
        x1,x2=x[:,0:length],x[:,length::]
        x1=self.layer1(x1)
        x2=self.layer2(x2)
        x_add=self.aux1(x1+x2)
        x_multi=self.aux2(x1*x2)
        return x_add+x_multi

class Tower(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Tower, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(input_size, hidden_size),nn.BatchNorm1d(hidden_size),nn.ReLU(),nn.Dropout(0.5))
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        # out = self.sigmoid(out)
        return out

class MMOE(nn.Module):
    def __init__(self, input_size, num_experts, experts_out, experts_hidden, towers_hidden,towers_out, tasks,pre_type=True):
        super(MMOE, self).__init__()
        self.input_size = input_size
        self.num_experts = num_experts
        self.experts_out = experts_out
        self.experts_hidden = experts_hidden
        self.towers_hidden = towers_hidden
        self.tasks = tasks
        self.towers_out=towers_out

        self.softmax = nn.Softmax(dim=1)

        self.experts = nn.ModuleList([Expert_MM(self.input_size, self.experts_out, self.experts_hidden)  for i in range(self.num_experts)])
        self.w_gates_cell=nn.ParameterList([nn.Parameter(torch.randn(927, num_experts), requires_grad=True) for i in range(self.tasks)])
        self.w_gates = nn.ParameterList([nn.Parameter(torch.randn(input_size, num_experts), requires_grad=True) for i in range(self.tasks)])
        self.w_gates_unli=nn.ModuleList([Expert(self.input_size, num_experts,self.experts_hidden) for i in range(self.tasks)])
        self.towers = nn.ModuleList([Tower(self.experts_out, self.towers_out, self.towers_hidden) for i in range(self.tasks)])
        self.pre_type=pre_type
    def forward(self, x):
        experts_o = [e(x) for e in self.experts]
        experts_o_tensor = torch.stack(experts_o)
        gate_type=2
        if gate_type==0:
            gates_o = [self.softmax(x @ g) for g in self.w_gates]
        elif gate_type==1:
            gates_o = [self.softmax(x[:,-927::] @ g) for g in self.w_gates_cell]
        elif gate_type==2:
            gates_o=[self.softmax(g(x)) for g in self.w_gates_unli]
        tower_input = [g.t().unsqueeze(2).expand(-1, -1, self.experts_out) * experts_o_tensor for g in gates_o]
        tower_input = [torch.sum(ti, dim=0) for ti in tower_input]

        final_output = [t(ti) for t, ti in zip(self.towers, tower_input)]
        return torch.stack(final_output,axis=2).squeeze(1) if self.pre_type else final_output

class MMOE_v2(nn.Module):
    def __init__(self, input_size, num_experts, name_experts, experts_out, experts_hidden, towers_hidden,towers_out, tasks,pre_type=True):
        super(MMOE_v2, self).__init__()
        self.input_size = input_size
        self.num_experts = num_experts
        self.experts_out = experts_out
        self.experts_hidden = experts_hidden
        self.towers_hidden = towers_hidden
        self.tasks = tasks
        self.towers_out=towers_out

        self.softmax = nn.Softmax(dim=1)
        self.experts=[]
        for name in name_experts:
            if name=='S':
                self.experts.append(Expert(self.input_size['S'], self.experts_out, self.experts_hidden))
            elif name=='M':
                self.experts.append(Expert_MM(self.input_size['M'], int(self.experts_out), self.experts_hidden))
            elif name=='B':
                self.experts.append(Expert_BI(self.input_size['M'], int(self.experts_out), self.experts_hidden))
        self.experts=nn.ModuleList(self.experts)
        # self.experts = nn.ModuleList([Expert(self.input_size, int(self.experts_out), self.experts_hidden) if i=='S' else Expert_MM(self.input_size, self.experts_out, self.experts_hidden) for i in name_experts])
        self.w_gates_cell=nn.ParameterList([nn.Parameter(torch.randn(927, num_experts), requires_grad=True) for i in range(self.tasks)])
        self.w_gates = nn.ParameterList([nn.Parameter(torch.randn(input_size['M'], num_experts), requires_grad=True) for i in range(self.tasks)])
        self.w_gates_unli=nn.ModuleList([Expert(self.input_size['M'], num_experts,self.experts_hidden) for i in range(self.tasks)])
        self.towers = nn.ModuleList([Tower(self.experts_out, self.towers_out, self.towers_hidden) for i in range(self.tasks)])
        self.pre_type=pre_type
    def forward(self, x):
        # experts_o = [e(X) for e,X in zip(self.experts,x)]
        experts_o = [e(x) for e in self.experts]
        experts_o_tensor = torch.stack(experts_o)
        gate_type=2
        if gate_type==0:
            gates_o = [self.softmax(x @ g) for g in self.w_gates]
        elif gate_type==1:
            gates_o = [self.softmax(x[:,-927::] @ g) for g in self.w_gates_cell]
        elif gate_type==2:
            # gates_o=[self.softmax(g(x[0])) for g in self.w_gates_unli]
            gates_o=[self.softmax(g(x)) for g in self.w_gates_unli]
        tower_input = [g.t().unsqueeze(2).expand(-1, -1, self.experts_out) * experts_o_tensor for g in gates_o]
        tower_input = [torch.sum(ti, dim=0) for ti in tower_input]

        final_output = [t(ti) for t, ti in zip(self.towers, tower_input)]
        return torch.stack(final_output,axis=2).squeeze(1) if self.pre_type else final_output

