import torch
import pennylane as qml
from torch import nn, Tensor
import numpy as np
import math

class VQC_Combined(nn.Module):
    def __init__(self, agents, q_value_indices, observation_length: int, num_layers: int, action_space: int):
        super(VQC_Combined, self).__init__()
        # layers and qubits
        # For Amplitude Embedding
        # self.num_qubits = max(math.ceil(math.log(observation_length, 2)), action_space)
        # For Basis Embedding
        self.agents = agents
        self.q_value_indices = q_value_indices
        self.num_qubits = max(2*observation_length, 2*action_space)
        
        self.num_layers = num_layers
        self.observation_length = observation_length

        # actions space
        self.action_space = 2*action_space
        self.device = qml.device("default.qubit", wires=self.num_qubits)
        self.qnode = qml.QNode(self.circuit, self.device, interface="torch")

        self.weights = nn.Parameter(
            torch.randn(num_layers, self.num_qubits, 3)
        )
        # self.expected_val_scaling = nn.Parameter(
        #     torch.ones(self.action_space, dtype=float)
        # )

        self.expected_val_scaling = {name : nn.Parameter(
            torch.ones(self.action_space // 2, dtype=float)
        ) for name in self.agents} 
        

        self.optimizer = {name : torch.optim.Adam([{"params": self.weights},{"params": self.expected_val_scaling[name], "lr": 0.1}], lr=0.001) for name in self.agents}
        
    def circuit(self, weights, x1, x2):
        qml.BasisEmbedding(features=x1, wires=range(self.observation_length))
        qml.BasisEmbedding(features=x2, wires=range(self.observation_length, 2*self.observation_length))

        for i in range(self.num_layers):
            for j in range(self.num_qubits):
                qml.Rot(*weights[i, j], wires=j)

            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[2, 3])

            # experiment_cnot_run01
            #qml.CNOT(wires=[1, 2])
            
            # experiment_cnot_run02
            #qml.CNOT(wires=[1, 2])
            #qml.CNOT(wires=[0, 3])

            # experiment_cnot_cycle_run03
            # qml.CNOT(wires=[0, 1])
            # qml.CNOT(wires=[0, 2])
            # qml.CNOT(wires=[0, 3])
            # qml.CNOT(wires=[1, 2])
            # qml.CNOT(wires=[1, 3])
            # qml.CNOT(wires=[2, 3])

            # experiment_cnot_ring_run04
            # qml.CNOT(wires=[0, 1])
            # qml.CNOT(wires=[1, 2])
            # qml.CNOT(wires=[2, 3])
            # qml.CNOT(wires=[3, 0])

            # experiment_cnot_layers_run05
            # qml.CNOT(wires=[0, 1])
            # qml.CNOT(wires=[2, 3])

            # qml.CNOT(wires=[1, 2])
            # qml.CNOT(wires=[3, 0])

            # experiment_cnot_hierachy_run06
            # qml.CNOT(wires=[0, 1])
            # qml.CNOT(wires=[0, 2])
            # qml.CNOT(wires=[2, 3])

            # experiment_cnot_dymamic_layer_run07
            # qml.CNOT(wires=[0, 1])
            # qml.CNOT(wires=[2, 3])

            # qml.CNOT(wires=[1, 2])
            # qml.CNOT(wires=[0, 3])

            # experiment_cnot_multi_control_toffoli_run08
            # qml.Toffoli(wires=[0, 1, 2])  
            # qml.CNOT(wires=[2, 3])

            # experiment_cnot_random_run09
            # qml.CNOT(wires=[0, 2])
            # qml.CNOT(wires=[1, 3])
            # qml.CNOT(wires=[3, 0])
            # qml.CNOT(wires=[2, 1])

            # experiment_cnot_parallel_run10
            # qml.CNOT(wires=[0, 1])
            # qml.CNOT(wires=[2, 3])

            # experiment_cnot_layer_skip_run11
            # qml.CNOT(wires=[0, 2])
            # qml.CNOT(wires=[1, 3])

            # qml.CNOT(wires=[0, 1])
            # qml.CNOT(wires=[2, 3])



        return [qml.expval(qml.PauliZ(i)) for i in range(self.action_space)]
    
    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        # Ensure x is the correct shape expected by the quantum circuit
        # if len(x.shape) == 1:
        #    x = x.unsqueeze(0)
        # x = torch.cat((x,torch.zeros(x.size())), dim=1)

    
        q_vals_agent = [torch.stack(self.scale(self.qnode(self.weights, i, j))) for i, j in torch.stack((x1, x2), dim=1)]
        q_vals = torch.stack(q_vals_agent)

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
        for agent, q_index in self.q_value_indices.items():
            q_index *= 2
            expected_vals[q_index] = expected_vals[q_index] * self.expected_val_scaling[agent][0]
            expected_vals[q_index+1] = expected_vals[q_index+1] * self.expected_val_scaling[agent][1]

        
        return expected_vals