from mlagents.trainers.torch.distributions import MultiCategoricalDistribution
import torch
import torch.nn as nn
import numpy as np
from mlagents_envs.environment import ActionTuple, UnityEnvironment


class WaveNetwork(nn.Module):
    def __init__(self, hidden_layer=512):
        super(WaveNetwork, self).__init__()
        self.dense = nn.Sequential(
            torch.nn.Linear(64, hidden_layer),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_layer, 2),
            torch.nn.Softmax(dim=1),
        )
        self.memory_size_vector = torch.nn.Parameter(
            torch.Tensor([0]), requires_grad=False
        )
        self.version_number = torch.nn.Parameter(torch.Tensor([3]), requires_grad=False)
        self.discrete_act_size_vector = torch.nn.Parameter(
            torch.Tensor([2]), requires_grad=False
        )

    def forward(self, obs1, obs2, action_masks=None):
        inp = torch.cat((obs2, obs1), dim=1)
        out = self.dense(inp)
        out = torch.argmax(out)

        export_out = [
            self.version_number,
            self.memory_size_vector,
            out.reshape((-1, 1, 1)),
            self.discrete_act_size_vector,
        ]
        return tuple(export_out)


def save_wave_model(hidden_layer, load_path, save_path):
    actor = WaveNetwork(hidden_layer)
    actor.dense.load_state_dict(torch.load(load_path))
    torch.onnx.export(
        actor,
        (
            torch.Tensor(np.zeros((1, 62))),
            torch.Tensor(np.zeros((1, 2))),
            torch.Tensor(np.zeros((1, 2))),
        ),
        save_path,
        opset_version=9,
        input_names=["obs_0", "obs_1", "action_masks"],
        output_names=[
            "version_number",
            "memory_size",
            "discrete_actions",
            "discrete_action_output_shape",
        ],
        dynamic_axes={
            "obs_0": {0: "batch_size"},
            "obs_1": {0: "batch_size"},
            "discrete_actions": {0: "batch_size"},
        },
    )
