import torch
import pennylane as qml
from torch import nn, Tensor
import matplotlib.pyplot as plt
import numpy as np
import math

class VQC_Combined(nn.Module):
    def __init__(self, agents, agent_order, observation_length: int, num_layers: int, action_space: int):
        super(VQC_Combined, self).__init__()
        # layers and qubits
        # For Amplitude Embedding
        # self.num_qubits = max(math.ceil(math.log(observation_length, 2)), action_space)
        # For Basis Embedding
        self.agents = agents
        self.agent_order = agent_order
        self.num_qubits = max(2*observation_length, 2*action_space)

        for agent, index in self.agent_order.items():
            if index == 0:
                wire_first_agent = [(agent, wire) for wire in range(0, self.num_qubits // 2)]
            elif index == 1:
                wire_second_agent = [(agent, wire) for wire in range(0, self.num_qubits // 2)]
        
        self.wire_assignment = wire_first_agent + wire_second_agent

        self.num_layers = num_layers
        self.observation_length = observation_length

        # actions space
        self.action_space = 2*action_space
        self.device = qml.device("default.qubit", wires=self.num_qubits)
        self.qnode = qml.QNode(self.circuit, self.device, interface="torch")

        # self.weights = nn.Parameter(
        #     torch.randn(num_layers, self.num_qubits // 2, 3)
        # )

        self.weights = {name : nn.Parameter(
            torch.randn(num_layers, self.num_qubits // 2, 3)
        ) for name in self.agents}

        # self.expected_val_scaling = nn.Parameter(
        #     torch.ones(self.action_space, dtype=float)
        # )

        self.expected_val_scaling = {name : nn.Parameter(
            torch.ones(self.action_space // 2, dtype=float)
        ) for name in self.agents} 
        

        self.optimizer = {name : torch.optim.Adam([{"params": self.weights[name]},{"params": self.expected_val_scaling[name], "lr": 0.1}], lr=0.001) for name in self.agents}
        
    def circuit(self, weights, x1, x2):
        # #experiment_increased_one_site_coop_run1_
        # qml.BasisEmbedding(features=x1, wires=range(self.observation_length))
        # qml.BasisEmbedding(features=x2, wires=range(self.observation_length, 2 * self.observation_length))

        # # # Bell-Zustand für Kooperation
        # qml.Hadamard(wires=0)
        # qml.CNOT(wires=[0, 2])

        # # # Verstärkter kooperativer Zustand
        # qml.RY(np.pi / 3, wires=2)
        # qml.CNOT(wires=[2, 3])


        qml.BasisEmbedding(features=x1, wires=range(self.observation_length))
        qml.BasisEmbedding(features=x2, wires=range(self.observation_length, 2*self.observation_length))
        
        for i in range(self.num_layers):
            for j in range(self.num_qubits):
                agent, agent_wire_index = self.wire_assignment[j]
                qml.Rot(*weights[agent][i][agent_wire_index], wires=j)


            # BASELINE
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[2, 3])

            # experiment_cnot_run03_
            qml.CNOT(wires=[0, 2])
            qml.CNOT(wires=[0, 3])

            # experiment_cnot_run04_
            # qml.CNOT(wires=[0, 2])
            # qml.CNOT(wires=[1, 3])

            # experiment_cnot_run05_
            # qml.CNOT(wires=[0, 2])  # Control: Agent 1 wire 0, Target: Agent 2 wire 2
            # qml.CNOT(wires=[1, 3])  # Control: Agent 1 wire 1, Target: Agent 2 wire 3

            # qml.CNOT(wires=[2, 0])  # Control: Agent 2 wire 2, Target: Agent 1 wire 0
            # qml.CNOT(wires=[3, 1])  # Control: Agent 2 wire 3, Target: Agent 1 wire 1

            # experiment_cnot_run06_
            # qml.CNOT(wires=[0, 2])
            # qml.CNOT(wires=[1, 3])

            # experiment_cnot_run07_
            # Multi-qubit entanglement using Toffoli gates
            # qml.Toffoli(wires=[0, 1, 2])  # Agent 1 controls Agent 2
            # qml.Toffoli(wires=[2, 3, 0])  # Agent 2 controls Agent 1


            # experiment_cnot_run08_
            # qml.SWAP(wires=[0, 2])  # Swap between Agent 1 and Agent 2
            # qml.SWAP(wires=[1, 3])  # Swap between Agent 1 and Agent 2

            # experiment_cnot_run09_
            # qml.CNOT(wires=[0, 2])  # Qubit von Agent 1 beeinflusst Agent 2
            # qml.CNOT(wires=[1, 3])  # Qubit von Agent 1 beeinflusst Agent 2
            # qml.CNOT(wires=[2, 0])  # Qubit von Agent 2 beeinflusst Agent 1
            # qml.CNOT(wires=[3, 1])  # Qubit von Agent 2 beeinflusst Agent 1

            # experiment_cnot_run10_
            # GHZ-Zustand über alle Qubits
            # qml.Hadamard(wires=0)
            # qml.CNOT(wires=[0, 1])
            # qml.CNOT(wires=[0, 2])
            # qml.CNOT(wires=[0, 3])





            
            
            # experiment_cnot_run02
            # qml.CNOT(wires=[1, 2])
            # qml.CNOT(wires=[0, 3])

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
        for agent, q_index in self.agent_order.items():
            q_index *= 2
            expected_vals[q_index] = expected_vals[q_index] * self.expected_val_scaling[agent][0]
            expected_vals[q_index+1] = expected_vals[q_index+1] * self.expected_val_scaling[agent][1]

        
        return expected_vals