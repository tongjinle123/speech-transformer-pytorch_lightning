import torch as t
from src.model.modules.gelu import Gelu


class JoinNet(t.nn.Module):
    def __init__(self, encoder_dim, lm_dim, model_size, vocab_size):
        super(JoinNet, self).__init__()
        self.linear = t.nn.Sequential(
            t.nn.Linear(in_features=encoder_dim+lm_dim, out_features=model_size, bias=True),
            Gelu(),
            t.nn.Linear(in_features=model_size, out_features=vocab_size, bias=True)
        )
        t.nn.init.xavier_normal_(self.linear[0].weight)
        t.nn.init.xavier_normal_(self.linear[2].weight)

    def forward(self, encoder_output, lm_output):
        # encoder_output B, Le, He
        # lm_output B, Ll, Hl
        batch_size, encoder_length, encoder_dim = encoder_output.size()
        batch_size, lm_length, lm_dim = lm_output.size()
        encoder_output = encoder_output.unsqueeze(2).repeat(1, 1, lm_length, 1)
        lm_output = lm_output.unsqueeze(1).repeat(1, encoder_length, 1, 1)
        net = t.cat([encoder_output, lm_output], dim=-1)
        net = self.linear(net)
        return net