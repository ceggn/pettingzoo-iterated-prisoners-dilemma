import torch
import pennylane as qml
from torch import nn, Tensor
import numpy as np
import math

class VQC(nn.Module):
    def __init__(self, observation_length: int, num_layers: int, action_space: int):
        super(VQC, self).__init__()
        # layers and qubits
        # For Amplitude Embedding
        # self.num_qubits = max(math.ceil(math.log(observation_length, 2)), action_space)
        # For Basis Embedding
        self.num_qubits = max(observation_length, action_space)
        
        self.num_layers = num_layers
        self.observation_length = observation_length

        # actions space
        self.action_space = action_space
        self.device = qml.device("default.qubit", wires=self.num_qubits)
        self.qnode = qml.QNode(self.circuit, self.device, interface="torch")

        self.weights = nn.Parameter(
            torch.randn(num_layers, self.num_qubits, 3)
        )
        self.expected_val_scaling = nn.Parameter(
            torch.ones(action_space, dtype=float)
        )
        

        self.optimizer = torch.optim.Adam([{"params": self.weights},{"params": self.expected_val_scaling, "lr": 0.1}], lr=0.001)
        
    def circuit(self, weights, x):
        for i in range(self.num_layers):
            #qml.AmplitudeEmbedding(features=x, wires=range(self.num_qubits), pad_with=0, normalize=True)
            qml.BasisEmbedding(features=x, wires=range(self.observation_length))
            for j in range(self.num_qubits):
                qml.Rot(*weights[i, j], wires=j)
            for j in range(self.num_qubits - 1):
                qml.CNOT(wires=[j, j+1])

        return [qml.expval(qml.PauliZ(i)) for i in range(self.action_space)]
    
    def forward(self, x: Tensor) -> Tensor:
        # Ensure x is the correct shape expected by the quantum circuit
        # if len(x.shape) == 1:
        #    x = x.unsqueeze(0)
        # x = torch.cat((x,torch.zeros(x.size())), dim=1)
    
        q_vals = [torch.stack(self.scale(self.qnode(self.weights, i))) for i in x]
        q_vals = torch.stack(q_vals)
        # q_vals = self.qnode(self.weights, x)
        # Convert q_vals (which is a list of tensors) to a single torch tensor
        #q_vals = torch.stack(q_vals, dim=1)
        
        # x = tensor([[a,b],[c,d]])
        # [z,w], [x,y]
        # [tensor([z,w]), tensor([x,y])]
        # tensor([[z,w], [x,y]])

        return q_vals

    def scale(self, expected_vals):
        # Scaling expected_vals from (-1,1) to (0,1)
        for i, a in enumerate(expected_vals):
            expected_vals[i] = (a + 1) / 2
            expected_vals[i] = expected_vals[i] * self.expected_val_scaling[i]
        
        
        return expected_vals