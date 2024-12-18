import torch

class MLP(torch.nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(args.feature_dim, args.sum_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(args.sum_hidden_dim, args.sum_output_dim),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        out = self.layers(x)
        return out