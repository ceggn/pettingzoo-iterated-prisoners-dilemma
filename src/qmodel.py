import math

import numpy as np

#import pennylane as qml
import torch as T

# Set default torch tensor type to Float64 to circumvent issue with amplitude encoding normalization
T.set_default_dtype(T.float64)

class qmodel():
    def __init__(self, input_size, output_size, learning_rate) -> None:
        self.output_size = output_size
        self.wires = max(math.ceil(math.log(input_size, 2)), int(output_size))
        self.num_layers = 1

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.weights = nn.Parameter(T.randint(0, 360, size=(self.num_layers, self.wires, 3), dtype=T.float64, device=self.device), requires_grad=True)
        self.optimizer = T.optim.SGD([self.weights], lr=learning_rate)
        self.criterion = nn.HuberLoss()

    def variational_layer(self, weights):
        for i in range(self.wires):
            qml.Rot(weights[i, 0], weights[i, 1], weights[i, 2], wires=i)
    
    def forward(self, observation):
        @qml.qnode(self.device('default.qubit', wires=self.wires, shots=1000))
        def circuit(weights, features):
            qml.templates.AmplitudeEmbedding(features, wires=range(self.wires), pad_with=0.0, normalize=True)
            for layer_weight in weights:
                self.variational_layer(layer_weight)
            return [qml.expval(qml.PauliZ(i)) for i in range(self.output_size)]
        
        return T.stack([T.stack(circuit(self.weights, i)) for i in observation]).to(self.device)
    
    def show(self, x):
        qml.draw(circuit)(self.weights, x)
    
    def save_checkpoint(self, checkpoint_path, filename="model") -> None:
        T.save(self.state_dict(), T.path.join(checkpoint_path, filename))
    
    def load_checkpoint(self, checkpoint_path, filename) -> None:
        self.load_state_dict(T.load(T.path.join(checkpoint_path, filename)))
