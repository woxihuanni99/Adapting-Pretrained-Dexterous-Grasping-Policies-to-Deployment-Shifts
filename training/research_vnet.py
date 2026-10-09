"""Grasp-success verifier V(o, a) for route 2 (sample-then-verify test-time selection).
Input: the frozen DemoGrasp planner's own actor input (proprio/object-init obs + point-cloud embedding), the planner
mean m and a candidate pre-tanh action u, encoded as [x_norm, tanh(m), tanh(u), 10*(tanh(u)-tanh(m))].
No friction input: the verifier must pick candidates that are robust under unknown friction.
An ensemble of E MLPs gives P(success); score = mean - beta * std (beta > 0 = pessimistic selection)."""
import torch
import torch.nn as nn


class Verifier(nn.Module):
    def __init__(self, in_dim, act_dim, hidden=256, ensemble=5, layers=3, dropout=0.0):
        super().__init__()
        d = in_dim + 3 * act_dim
        self.nets = nn.ModuleList()
        for _ in range(ensemble):
            mods, h = [], d
            for _ in range(layers):
                mods += [nn.Linear(h, hidden), nn.LayerNorm(hidden), nn.SiLU()] + ([nn.Dropout(dropout)] if dropout > 0 else [])
                h = hidden
            mods.append(nn.Linear(h, 1))
            self.nets.append(nn.Sequential(*mods))
        self.register_buffer("x_mean", torch.zeros(in_dim))
        self.register_buffer("x_std", torch.ones(in_dim))

    def features(self, x, m, u):
        x = (x.float() - self.x_mean) / self.x_std
        tm, tu = torch.tanh(m.float()), torch.tanh(u.float())
        return torch.cat([x, tm, tu, 10.0 * (tu - tm)], -1)

    def forward(self, x, m, u, members=None):
        f = self.features(x, m, u)
        nets = self.nets if members is None else [self.nets[i] for i in members]
        return torch.stack([n(f).squeeze(-1) for n in nets], 0)  # (E, B) logits

    @torch.no_grad()
    def score(self, x, m, u, beta=0.0):
        p = torch.sigmoid(self.forward(x, m, u))
        return p.mean(0) - beta * (p.std(0) if p.shape[0] > 1 else 0.0)


def save_verifier(v, path, meta):
    torch.save({"state_dict": v.state_dict(), "meta": meta}, str(path))


def load_verifier(path, device):
    d = torch.load(str(path), map_location=device)
    m = d["meta"]
    v = Verifier(m["in_dim"], m["act_dim"], m["hidden"], m["ensemble"], m["layers"], m.get("dropout", 0.0))
    v.load_state_dict(d["state_dict"])
    return v.to(device).eval(), m
