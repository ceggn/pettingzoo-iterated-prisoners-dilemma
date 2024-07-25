# vqc_model.py

import torch
import pennylane as qml
from torch import nn, Tensor
import numpy as np

class VQC(nn.Module):
    def __init__(self, num_qubits: int, num_layers: int, action_space: int):
        super(VQC, self).__init__()
        #layers and qubits
        self.num_qubits = num_qubits
        self.num_layers = num_layers

        # actions space
        self.action_space = action_space

        self.device = qml.device("default.qubit", wires=num_qubits)
        self.qnode = qml.QNode(self.circuit, self.device, interface="torch")

        self.weights = nn.Parameter(
            0.01 * torch.randn(num_layers, num_qubits, 3)
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        
    def circuit(self, weights, x):
        qml.templates.AmplitudeEmbedding(features=x, wires=range(self.num_qubits), normalize=True)
        for i in range(self.num_layers):
            for j in range(self.num_qubits):
                qml.Rot(*weights[i, j], wires=j)
            for j in range(self.num_qubits - 1):
                qml.CNOT(wires=[j, j+1])
        return [qml.expval(qml.PauliZ(i)) for i in range(self.action_space)]

    def forward(self, x: Tensor) -> Tensor:
        q_vals = self.qnode(self.weights, x)
        return torch.tensor(q_vals)

    def train_model(self, states, actions, rewards, next_states, dones, gamma):
        current_q_values = self(states).gather(1, actions)
        next_q_values = self(next_states).max(1)[0].detach().unsqueeze(1)
        target_q_values = rewards + (gamma * next_q_values * (1 - dones))
        loss = nn.functional.mse_loss(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()