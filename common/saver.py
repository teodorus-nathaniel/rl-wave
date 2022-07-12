import numpy as np
import torch
import torch.nn as nn


class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.memory_size_vector = torch.nn.Parameter(
            torch.Tensor([0]), requires_grad=False
        )
        self.version_number = torch.nn.Parameter(torch.Tensor([3]), requires_grad=False)
        self.discrete_act_size_vector = torch.nn.Parameter(
            torch.Tensor([2]), requires_grad=False
        )

    def forward(self, obs1, obs2, _action_masks=None):
        inp = torch.cat((obs2, obs1), dim=1)
        out = self.model(inp)
        if isinstance(out, list) or isinstance(out, tuple):
            out = out[0].probs

        out = torch.argmax(out)

        export_out = [
            self.version_number,
            self.memory_size_vector,
            out.reshape((-1, 1, 1)),
            self.discrete_act_size_vector,
        ]
        return tuple(export_out)


def save_wave_model(model, load_path, save_path):
    actor = ModelWrapper(model)
    actor.model.load_state_dict(torch.load(load_path))
    torch.onnx.export(
        actor,
        (
            torch.Tensor(np.zeros((1, 124))),
            torch.Tensor(np.zeros((1, 3))),
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
