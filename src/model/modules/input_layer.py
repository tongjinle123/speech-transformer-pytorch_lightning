import torch as t
import fairseq as fs


class InputLayer(t.nn.Module):
    """
    input spec resize layer combine highway layer
    """
    def __init__(self, input_size, model_size, dropout):
        super(InputLayer, self).__init__()
        self.input_linear = t.nn.Linear(input_size, model_size)
        self.input_layer_norm = t.nn.LayerNorm(model_size)
        # self.input_highway = fs.modules.Highway(input_dim=model_size, num_layers=2)
        self.input_dropout = t.nn.Dropout(dropout)
        t.nn.init.xavier_normal_(self.input_linear.weight)

    def forward(self, wave_feature):
        net = self.input_linear(wave_feature)
        net = t.nn.functional.relu(net)
        # net = self.input_layer_norm(net)
        net = self.input_dropout(net)
        # net = self.input_highway(net)
        return net