import torch
from torch import nn
import torch.nn.utils as utils
from torch.optim import Adam
import torch.nn.functional as F
from torch.autograd import Variable

from torch.distributions import Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

class Model(torch.nn.Module):
    """
    The network for ES implementation.
    """
    def __init__(self,input_dim:int, hidden_dim:[int, int, int], output_dim:[int, int, int, int]):
        super(Model, self).__init__()
        self.input = nn.Linear(input_dim, hidden_dim[0])
        self.hidden0 = nn.Linear(hidden_dim[0], hidden_dim[1])
        
        self.hidden_type0 = nn.Linear(hidden_dim[1], hidden_dim[2])
        self.hidden_type1 = nn.Linear(hidden_dim[1], hidden_dim[2])
        self.hidden_move0 = nn.Linear(hidden_dim[1], hidden_dim[2])
        self.hidden_move1 = nn.Linear(hidden_dim[1], hidden_dim[2])
        self.hidden_move2 = nn.Linear(hidden_dim[1], hidden_dim[2])
        self.hidden_move3 = nn.Linear(hidden_dim[1], hidden_dim[2])
        
        self.output_type0 = nn.Linear(hidden_dim[2], output_dim[0])
        self.output_type1 = nn.Linear(hidden_dim[2], output_dim[1])
        self.output_move0 = nn.Linear(hidden_dim[2], output_dim[2])
        self.output_move1 = nn.Linear(hidden_dim[2], output_dim[3])
        self.output_move2 = nn.Linear(hidden_dim[2], output_dim[3])
        self.output_move3 = nn.Linear(hidden_dim[2], output_dim[3])
        
    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.hidden0(x))
        
        x_type0 = F.relu(self.hidden_type0(x))
        x_type1 = F.relu(self.hidden_type1(x))
        x_move0 = F.relu(self.hidden_move0(x))
        x_move1 = F.relu(self.hidden_move1(x))
        x_move2 = F.relu(self.hidden_move2(x))
        x_move3 = F.relu(self.hidden_move3(x))
        
        x_type0 = self.output_type0(x_type0)
        x_type1 = self.output_type1(x_type1)
        x_move0 = self.output_move0(x_move0)
        x_move1 = self.output_move1(x_move1)
        x_move2 = self.output_move2(x_move2)
        x_move3 = self.output_move3(x_move3)
        
        return(x_type0, x_type1, x_move0, x_move1, x_move2, x_move3)
    
class Actor(nn.Module):
    """
    The actor network for A2C implementation.
    """
    def __init__(self,input_dim:int, hidden_dim:[int, int, int], output_dim:[int, int, int, int]):
        super(Actor, self).__init__()
        self.input = nn.Linear(input_dim, hidden_dim[0])
        self.hidden0 = nn.Linear(hidden_dim[0], hidden_dim[1])
        
        self.hidden_type0 = nn.Linear(hidden_dim[1], hidden_dim[2])
        self.hidden_type1 = nn.Linear(hidden_dim[1], hidden_dim[2])
        self.hidden_move0 = nn.Linear(hidden_dim[1], hidden_dim[2])
        self.hidden_move1 = nn.Linear(hidden_dim[1], hidden_dim[2])
        self.hidden_move2 = nn.Linear(hidden_dim[1], hidden_dim[2])
        self.hidden_move3 = nn.Linear(hidden_dim[1], hidden_dim[2])
        
        self.output_type0 = nn.Linear(hidden_dim[2], output_dim[0])
        self.output_type1 = nn.Linear(hidden_dim[2], output_dim[1])
        self.output_move0 = nn.Linear(hidden_dim[2], output_dim[2])
        self.output_move1 = nn.Linear(hidden_dim[2], output_dim[3])
        self.output_move2 = nn.Linear(hidden_dim[2], output_dim[3])
        self.output_move3 = nn.Linear(hidden_dim[2], output_dim[3])
    
    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.hidden0(x))
        
        x_type0 = F.relu(self.hidden_type0(x))
        x_type1 = F.relu(self.hidden_type1(x))
        x_move0 = F.relu(self.hidden_move0(x))
        x_move1 = F.relu(self.hidden_move1(x))
        x_move2 = F.relu(self.hidden_move2(x))
        x_move3 = F.relu(self.hidden_move3(x))
        
        x_type0 = self.output_type0(x_type0)
        x_type1 = self.output_type1(x_type1)
        x_move0 = self.output_move0(x_move0)
        x_move1 = self.output_move1(x_move1)
        x_move2 = self.output_move2(x_move2)
        x_move3 = self.output_move3(x_move3)
        
        dist_type0 = Categorical(F.softmax(x_type0, dim=-1))
        dist_type1 = Categorical(F.softmax(x_type1, dim=-1))
        dist_move0 = Categorical(F.softmax(x_move0, dim=-1))
        dist_move1 = Categorical(F.softmax(x_move1, dim=-1))
        dist_move2 = Categorical(F.softmax(x_move2, dim=-1))
        dist_move3 = Categorical(F.softmax(x_move3, dim=-1))
        
        return(dist_type0, dist_type1, dist_move0, dist_move1, dist_move2, dist_move3)
    

class Critic(nn.Module):
    """
    The critic network for A2C implementation.
    """
    def __init__(self,input_dim:int, hidden_dim:[int, int, int], output_dim:[int, int, int, int]):
        super(Critic, self).__init__()
        self.input = nn.Linear(input_dim, hidden_dim[0])
        self.hidden0 = nn.Linear(hidden_dim[0], hidden_dim[1])
        
        self.hidden_type0 = nn.Linear(hidden_dim[1], hidden_dim[2])
        self.hidden_type1 = nn.Linear(hidden_dim[1], hidden_dim[2])
        self.hidden_move0 = nn.Linear(hidden_dim[1], hidden_dim[2])
        self.hidden_move1 = nn.Linear(hidden_dim[1], hidden_dim[2])
        self.hidden_move2 = nn.Linear(hidden_dim[1], hidden_dim[2])
        self.hidden_move3 = nn.Linear(hidden_dim[1], hidden_dim[2])
        
        self.output_type0 = nn.Linear(hidden_dim[2], 1)
        self.output_type1 = nn.Linear(hidden_dim[2], 1)
        self.output_move0 = nn.Linear(hidden_dim[2], 1)
        self.output_move1 = nn.Linear(hidden_dim[2], 1)
        self.output_move2 = nn.Linear(hidden_dim[2], 1)
        self.output_move3 = nn.Linear(hidden_dim[2], 1)
        
    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.hidden0(x))
        
        x_type0 = F.relu(self.hidden_type0(x))
        x_type1 = F.relu(self.hidden_type1(x))
        x_move0 = F.relu(self.hidden_move0(x))
        x_move1 = F.relu(self.hidden_move1(x))
        x_move2 = F.relu(self.hidden_move2(x))
        x_move3 = F.relu(self.hidden_move3(x))
        
        value_type0 = self.output_type0(x_type0)
        value_type1 = self.output_type1(x_type1)
        value_move0 = self.output_move0(x_move0)
        value_move1 = self.output_move1(x_move1)
        value_move2 = self.output_move2(x_move2)
        value_move3 = self.output_move3(x_move3)
        
        return(value_type0, value_type1, value_move0, value_move1, value_move2, value_move3)