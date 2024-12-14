

class MLP(nn.Module):
    def __init__(self, args):
        super(SummerizerMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(args.feature_dim, args.sum_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.sum_hidden_dim, args.sum_output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.layers(x)
        return out