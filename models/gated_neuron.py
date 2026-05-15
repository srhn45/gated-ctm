import torch
import torch.nn as nn
from models.modules import SuperLinear

class GatedNeuronBlock(nn.Module):
    def __init__(
        self,
        in_dims,
        out_dims,
        N,
        do_norm,
        dropout,
        activation="glu",   # "glu", "sigmoid", "tanh"
        gating=None         # None, "sigmoid", "tanh"
    ):
        super().__init__()

        # main branch
        main_out = 2 * out_dims if activation == "glu" else out_dims
        self.main = SuperLinear(
            in_dims=in_dims,
            out_dims=main_out,
            N=N,
            do_norm=do_norm,
            dropout=dropout
        )

        if activation == "glu":
            self.activation = nn.GLU()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"unknown activation {activation}")

        # optional gate branch
        if gating is not None:
            self.gate = SuperLinear(
                in_dims=in_dims,
                out_dims=out_dims,
                N=N,
                do_norm=do_norm,
                dropout=dropout
            )

            if gating == "sigmoid":
                self.gate_act = nn.Sigmoid()
            elif gating == "tanh":
                self.gate_act = nn.Tanh()
            else:
                raise ValueError(f"unknown gating {gating}")
        else:
            self.gate = None

    def forward(self, x):
        y = self.activation(self.main(x))

        if self.gate is not None:
            g = self.gate_act(self.gate(x)).view_as(y)
            y = y * g

        return y