import torch
import torch.nn as nn
import torch.nn.functional as F



class Elman(nn.Module):
    def __init__(self, insize=300, outsize=300, hsize=300):
        super(Elman, self).__init__()
        self.lin1 = nn.Linear(insize + hsize, hsize)
        self.lin2 = nn.Linear(hsize, outsize)
    def forward(self, x, hidden=None):
        b, t, e = x.size()
        if hidden is None:
            hidden = torch.zeros(b, e, dtype=torch.float)

        outs = []
        for i in range(t):
            inp = torch.cat([x[:, i, :], hidden], dim=1)
            inp1 = self.lin1(inp)
            hidden = F.relu(inp1)
            # hidden = torch.tanh(self.lin1(inp))
            # hidden = F.dropout(hidden, p=0.5)
            out = self.lin2(hidden)
            outs.append(out[:, None, :])

        return torch.cat(outs, dim=1), hidden