import torch
import pennylane as qml
from torch import nn, Tensor
import numpy as np
import math

class VQC(nn.Module):
    def __init__(self, num_qubits: int, num_layers: int, action_space: int):
        super(VQC, self).__init__()
        # layers and qubits
        self.num_qubits = max(math.ceil(math.log(num_qubits, 2)), action_space)
        self.num_layers = num_layers

        # actions space
        self.action_space = action_space
        self.device = qml.device("default.qubit", wires=self.num_qubits)
        self.qnode = qml.QNode(self.circuit, self.device, interface="torch")

        self.weights = nn.Parameter(
            torch.randn(num_layers, self.num_qubits, 3)
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        
    def circuit(self, weights, x):
        #qml.AmplitudeEmbedding(features=x, wires=range(self.num_qubits), pad_with=0, normalize=True)
        qml.BasisEmbedding(features=x, wires=range(self.num_qubits))
        for i in range(self.num_layers):
            for j in range(self.num_qubits):
                qml.Rot(*weights[i, j], wires=j)
            for j in range(self.num_qubits - 1):
                qml.CNOT(wires=[j, j+1])

        return [qml.expval(qml.PauliZ(i)) for i in range(self.action_space)]
    
    def forward(self, x: Tensor) -> Tensor:
        # Ensure x is the correct shape expected by the quantum circuit
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        x = torch.cat((x,torch.zeros(x.size())), dim=1)
    
        q_vals = torch.stack([torch.stack(self.qnode(self.weights, i)) for i in x])
        # q_vals = self.qnode(self.weights, x)
        # Convert q_vals (which is a list of tensors) to a single torch tensor
        #q_vals = torch.stack(q_vals, dim=1)
        
        return q_vals
