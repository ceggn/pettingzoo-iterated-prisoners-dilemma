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
                # qml.RY(weights[agent][i][agent_wire_index, 0], wires=j)
                # qml.RZ(weights[agent][i][agent_wire_index, 1], wires=j)



            # BASELINE
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[2, 3])

            # experiment_cnot_run03_
            # qml.CNOT(wires=[0, 2])
            # qml.CNOT(wires=[0, 3])

            # experiment_cnot_run04_
            # qml.CNOT(wires=[0, 2])
            # qml.CNOT(wires=[1, 3])

            # experiment_cnot_run05_
            # qml.CNOT(wires=[0, 2])  # Control: Agent 1 wire 0, Target: Agent 2 wire 2
            # qml.CNOT(wires=[1, 3])  # Control: Agent 1 wire 1, Target: Agent 2 wire 3

            # qml.CNOT(wires=[2, 0])  # Control: Agent 2 wire 2, Target: Agent 1 wire 0
            # qml.CNOT(wires=[3, 1])  # Control: Agent 2 wire 3, Target: Agent 1 wire 1

            # experiment_cnot_run06_
            # qml.CNOT(wires=[0, 3])
            # qml.CNOT(wires=[1, 2])

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

            # experiment_cnot_run10_ (ghz state)
            # GHZ-Zustand über alle Qubits
            # qml.Hadamard(wires=0)
            # qml.CNOT(wires=[0, 1])
            # qml.CNOT(wires=[0, 2])
            # qml.CNOT(wires=[0, 3])


            # experiment_cnot_run11_
            # qml.Hadamard(wires=0)
            # qml.CNOT(wires=[0, 1])  
            # qml.CNOT(wires=[0, 2])  
            # qml.CNOT(wires=[0, 3])
            # qml.CNOT(wires=[1, 2])
            # qml.CNOT(wires=[2, 3])


            # experiment_cnot_run12_
            # qml.Hadamard(wires=0)
            # qml.CNOT(wires=[0, 2])  
            # qml.CNOT(wires=[1, 3]) 

            # experiment_cnot_run13_
            # qml.Hadamard(wires=0)
            # qml.Hadamard(wires=2)
            # qml.CNOT(wires=[0, 2])  
            # qml.CNOT(wires=[1, 3])
            # qml.CNOT(wires=[1, 3])


            # experiment_cnot_run14_
            # qml.Hadamard(wires=0)
            # qml.CNOT(wires=[0, 2])  
            # qml.CNOT(wires=[1, 3]) 

            # experiment_cnot_run15_
            # qml.Hadamard(wires=0)
            # qml.CNOT(wires=[0, 1])
            # qml.CNOT(wires=[1, 2]) 
            # qml.CNOT(wires=[2, 3]) 

            # experiment_cnot_run16_
            # qml.Hadamard(wires=0)
            # qml.CNOT(wires=[0, 1])
            # qml.CNOT(wires=[1, 2]) 
            # qml.CNOT(wires=[2, 3])
            # qml.CNOT(wires=[3, 0]) 


            # experiment_cnot_run17_
            # qml.CNOT(wires=[0, 2])
            # qml.CNOT(wires=[0, 3]) 
            # qml.CNOT(wires=[1, 2])
            # qml.CNOT(wires=[1, 3]) 

            # # experiment_cnot_run18_
            # qml.Hadamard(wires=0)
            # qml.Hadamard(wires=2)

            # experiment_cnot_run19_
            # qml.Hadamard(wires=0)
            # qml.CNOT(wires=[0, 1])
            # qml.CRY(np.pi / 3, wires=[0, 2])
            # qml.CRY(np.pi / 3, wires=[1, 3])

            # # experiment_cnot_run_20_
            # qml.Toffoli(wires=[0, 1, 2])
            # qml.Toffoli(wires=[2, 3, 0])

            # experiment_cnot_run21_
            # bell
            # qml.Hadamard(wires=0)
            # qml.CNOT(wires=[0, 2])  # Agent 1 beeinflusst Agent 2

            # experiment_cnot_run22_
            # qml.SWAP(wires=[0, 2])  # Tauscht Qubit zwischen Agent 1 und Agent 2
            # qml.SWAP(wires=[1, 3])  # Tauscht zweites Qubit zwischen Agenten

            # # experiment_cnot_run23_
            # # GHZ-Zustand für starke Korrelationen
            # qml.Hadamard(wires=0)   
            # qml.CNOT(wires=[0, 1])  
            # qml.CNOT(wires=[0, 2])  
            # qml.CNOT(wires=[0, 3])       
            # # SWAP-Gatter für Informationsaustausch zwischen Agenten
            # qml.SWAP(wires=[0, 2])  
            # qml.SWAP(wires=[1, 3])  

            # experiment_cnot_run24_ (stronger GHZ state)
            # GHZ-Zustand für starke Korrelationen
            # qml.Hadamard(wires=0)   
            # qml.CNOT(wires=[0, 1])  
            # qml.CNOT(wires=[0, 2])  
            # qml.CNOT(wires=[0, 3])       
            # # SWAP-Gatter für Informationsaustausch zwischen Agenten
            # qml.CNOT(wires=[0, 2])  
            # qml.CNOT(wires=[1, 3]) 

            # #  Unidirectional Control for Forced Cooperation
            # # experiment_cnot_run25
            # qml.PauliX(wires=0)  # Force Agent 1 into defection
            # qml.CNOT(wires=[0, 2])  # Agent 1 enforces its decision onto Agent 2
            # qml.RZ(-np.pi / 4, wires=2)  # Introduce bias toward defection
            # qml.CNOT(wires=[0, 3])  # Further enforce defection onto Agent 2


            # experiment_cnot_run26
            # qml.PauliX(wires=0)  # Force Agent 1 into defection
            # qml.CNOT(wires=[0, 2])  # Agent 1 enforces its decision onto Agent 2
            # qml.RZ(-np.pi / 4, wires=2)  # Introduce bias toward defection
            # qml.CNOT(wires=[0, 3])  # Further enforce defection onto Agent 2

            # experiment_cnot_run27
            # qml.Hadamard(wires=0)  # Agent 1 starts in superposition
            # qml.CNOT(wires=[0, 2])  # Agent 1 influences Agent 2
            # qml.CZ(wires=[1, 3])  # Conditional phase shift for Agent 2’s second qubit
            # qml.RY(np.pi / 3, wires=2)  # Small cooperative bias for Agent 2
            # qml.RX(-np.pi / 4, wires=3)  # Introduce randomness for Agent 2

            # experiment_cnot_run28
            # qml.Hadamard(wires=0)  # Agent 1 explores cooperation
            # qml.CNOT(wires=[0, 2])  # Agent 1 entangles with Agent 2
            # qml.T(wires=2)  # Introduce a delay effect on Agent 2
            # qml.CNOT(wires=[2, 3])  # Pass the defection information forward
            # qml.SWAP(wires=[2, 3])  # Exchange qubits for delayed impact


            # experiment_cnot_run29
            # qml.Hadamard(wires=0)  # Agent 1 starts in superposition (neutral)
            # qml.CNOT(wires=[0, 2])  # Agent 1’s state influences Agent 2
            # qml.Toffoli(wires=[2, 3, 0])  # If Agent 2 defects, Agent 1 defects next round
            # qml.RY(np.pi / 6, wires=2)  # Small cooperative bias for Agent 2
            # qml.CNOT(wires=[3, 1])  # Agent 2 influences Agent 1’s second qubit


            # #  Bidirectional Control for Forced Cooperation
            # # experiment_cnot_run30
            # qml.CNOT(wires=[0, 2])  # Agent 1 beeinflusst Agent 2
            # qml.CNOT(wires=[2, 0])  # Agent 2 beeinflusst Agent 1
            # qml.CNOT(wires=[1, 3])  # Agent 1 beeinflusst zweiten Qubit von Agent 2
            # qml.CNOT(wires=[3, 1])  # Agent 2 beeinflusst zweiten Qubit von Agent 1

            # #experiment_cnot_run31
            # qml.CZ(wires=[0, 2])
            # qml.CZ(wires=[1, 3])

            # #experiment_cnot_run32
            # qml.CSWAP(wires=[0, 1, 2])

            # #experiment_cnot_run33
            qml.CRY(np.pi / 4, wires=[0, 2])
            qml.CRX(np.pi / 6, wires=[1, 3])

            # #experiment_cnot_run34
            # qml.SWAP(wires=[0, 2])
            # qml.SWAP(wires=[1, 3])

            # # Multi-Qubit Entanglement
            # # experiment_cnot_run35
            # qml.Toffoli(wires=[0, 2, 3])  # If both Q_0 (Agent 1 Cooperation) & Q_2 (Agent 2 Cooperation) are 1, flip Q_3 (Agent 2 Defection)


            # # experiment_cnot_run36
            # qml.Toffoli(wires=[1, 3, 0])  # If both Q_1 (Agent 1 Defection) & Q_3 (Agent 2 Defection) are 1, flip Q_0 (Agent 1 Cooperation)


            # #experiment_qft_run37
            # qml.QFT(wires=[0, 1, 2, 3])

            # #experiment_q_walk_run38
            # qml.Hadamard(wires=0)
            # qml.CZ(wires=[0, 1])
            # qml.CZ(wires=[2, 3])
            # qml.SWAP(wires=[0, 2])

            # #experiment_depolnoise_run39
            # qml.DepolarizingChannel(0.1, wires=0)

            # #experiment_bitflip_run40
            # qml.BitFlip(p=0.2, wires=1)
            # qml.PhaseFlip(p=0.3, wires=2)

            # #experiment_amplitude_damping_run41
            # qml.AmplitudeDamping(0.1, wires=3)


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