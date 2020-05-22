import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        scale_coeff = 3
        return x * (torch.tanh(F.softplus(scale_coeff*x)))


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


def gelu(x):
    scale_coeff = 3
    return 0.5 * x * (1 + torch.tanh(math.sqrt(math.pi / 2) * (scale_coeff*x + 0.044715 * scale_coeff* x ** 3)))


def sharkfin(input):
    '''
    Applies the sharkfin function element-wise:
    sharkfin(x) = tanh(e^{x}) * relu(-1, x) = tanh(e^{x}) * max(-1,x)
    From: https://github.com/digantamisra98/SharkFin/blob/master/torch/functional.py
    '''
    return torch.tanh(torch.exp(input)) * torch.clamp(input, min=-1)

def sineReLU(input, eps=0.01):
    """
    Applies the SineReLU activation function element-wise:
    .. math::
        SineReLU(x, \\epsilon) = \\left\\{\\begin{matrix} x , x > 0 \\\\ \\epsilon * (sin(x) - cos(x)), x \\leq  0 \\end{matrix}\\right.
    See additional documentation for :mod:`echoAI.Activation.Torch.sine_relu`.
    From: https://github.com/digantamisra98/Echo/blob/master/echoAI/Activation/Torch/functional.py
    """
    return (input > 0).float() * input + (input <= 0).float() * eps * (
        torch.sin(input) - torch.cos(input)
    )

def nl_relu(x, beta=1., inplace=False):
    """
    Applies the natural logarithm ReLU activation function element-wise:
    See additional documentation for :mod:`echoAI.Activation.Torch.nl_relu`.
    From: https://github.com/digantamisra98/Echo/blob/master/echoAI/Activation/Torch/functional.py
    """

    if inplace:
        return torch.log(F.relu_(x).mul_(beta).add_(1), out=x)
    else:
        return torch.log(1 + beta * F.relu(x))

class Ffn(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim=4*4096, output_dim=4096,
                 drop=0.0):
        super(Ffn, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.drop = nn.Dropout(p=drop)

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SteFfn(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim=4*4096, output_dim=4096,
                 drop=0.0, ste_layers_cnt=4):
        super(SteFfn, self).__init__()
        self.ste_fc = SteLin(input_dim, output_dim, drop, ste_layers_cnt)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.ste_fc(x)
        x = self.act1(x)
        x = self.fc2(x)
        return x


class Glu(nn.Module):
    def __init__(self, input_dim=4096, output_dim=4096):
        super(Glu, self).__init__()
        self.g1 = nn.Linear(input_dim, output_dim)
        self.act1 = nn.Sigmoid()
        self.fc1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.act1(self.g1(x)) * self.fc1(x)
        return x


class SteLin(nn.Module):
    def __init__(self, input_dim=4096, output_dim=4096, drop=0.5,
                 layers_cnt=4):
        super(SteLin, self).__init__()
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.layers_cnt = layers_cnt
        for i in range(self.layers_cnt):
            self.layers.append(nn.Linear(input_dim, output_dim))
            self.dropouts.append(nn.Dropout(p=drop))

    def forward(self, x):
        y = None
        for i in range(self.layers_cnt):
            if y is None:
                y = self.dropouts[i](self.layers[i](x))
            else:
                y = y + self.dropouts[i](self.layers[i](x))
        y = y / self.layers_cnt
        return y


class SteGlu(nn.Module):
    def __init__(self, input_dim=4096, output_dim=4096, drop=0.5,
                 ste_layers_cnt=4):
        super(SteGlu, self).__init__()
        self.g1 = nn.Linear(input_dim, output_dim)
        self.act1 = nn.Sigmoid()
        self.ste_fc = SteLin(input_dim, output_dim, drop, ste_layers_cnt)

    def forward(self, x):
        y = self.ste_fc(x)
        x = self.act1(self.g1(x)) * y
        return x


class SteGluFfn(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim=4*4096, output_dim=4096,
                 drop=0.5, ste_layers_cnt=4):
        super(SteGluFfn, self).__init__()
        self.glu1 = SteGlu(input_dim=input_dim,
                           output_dim=hidden_dim,
                           drop=0.5,
                           ste_layers_cnt=ste_layers_cnt)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.glu1(x)
        x = self.fc2(x)
        return x


class GluFfn(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim=4*4096, output_dim=4096):
        super(GluFfn, self).__init__()
        self.glu1 = Glu(input_dim=input_dim,
                        output_dim=hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.glu1(x)
        x = self.fc2(x)
        return x


class ResFfn(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim=4*4096, output_dim=4096,
                 drop=0.0):
        super(ResFfn, self).__init__()
        self.ffn1 = Ffn(input_dim=input_dim,
                        hidden_dim=hidden_dim,
                        output_dim=output_dim)
        self.res_drop = nn.Dropout(p=drop)

    def forward(self, x):
        x = x + self.res_drop(self.ffn1(x))
        return x


class ResGlu(nn.Module):
    def __init__(self, input_dim=4096, output_dim=4096,
                 drop=0.0):
        super(ResGlu, self).__init__()
        self.glu1 = Glu(input_dim=input_dim,
                        output_dim=output_dim)
        self.res_drop = nn.Dropout(p=drop)

    def forward(self, x):
        x = x + self.res_drop(self.glu1(x))
        return x


class GateOutFfn(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim=4*4096, output_dim=4096,
                 drop=0.0):
        super(GateOutFfn, self).__init__()
        self.ffn1 = Ffn(input_dim=input_dim,
                        hidden_dim=hidden_dim,
                        output_dim=output_dim)
        self.gfc = nn.Linear(input_dim+output_dim, output_dim)
        self.res_drop = nn.Dropout(p=drop)

    def forward(self, x):
        y = self.ffn1(x)
        g = torch.sigmoid(self.gfc(torch.cat((x, y), dim=1)))
        x = x + self.res_drop(g*y)
        return x


class DenseLayer(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim=4096, output_dim=4096,
                 drop=0.0, drop_pos='output', norm=None, highway_network=None,
                 cat_inout=False, hidden_act='relu'):
        super(DenseLayer, self).__init__()
        self.drop_pos = drop_pos
        self.norm = norm
        self.highway_network = highway_network
        self.cat_inout = cat_inout
        self.hidden_act = hidden_act

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(input_dim+hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(input_dim+2*hidden_dim, output_dim)

        if self.hidden_act == 'relu':
            self.act1 = nn.ReLU()
            self.act2 = nn.ReLU()
        elif self.hidden_act == 'gelu':
            self.act1 = gelu
            self.act2 = gelu
        elif self.hidden_act == 'mish':
            self.act1 = Mish()
            self.act2 = Mish()
        elif self.hidden_act == 'swish':
            self.act1 = Swish()
            self.act2 = Swish()
        elif self.hidden_act == 'sharkfin':
            self.act1 = sharkfin
            self.act2 = sharkfin
        elif self.hidden_act == 'sinrelu':
            self.act1 = sineReLU
            self.act2 = sineReLU
        elif self.hidden_act == 'nlrelu':
            self.act1 = nl_relu
            self.act2 = nl_relu

        self.drop1 = nn.Dropout(p=drop)
        self.drop2 = nn.Dropout(p=drop)
        if self.drop_pos == 'input':
            self.drop3 = nn.Dropout(p=drop)
        if self.norm == 'batch':
            self.norm1 = nn.BatchNorm1d(hidden_dim)
            self.norm2 = nn.BatchNorm1d(hidden_dim)
        elif self.norm == 'layer':
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.norm2 = nn.LayerNorm(hidden_dim)
        if self.highway_network == 'Wg(x)':
            self.dropg1 = nn.Dropout(p=drop)
            self.dropg2 = nn.Dropout(p=drop)
            self.wg = nn.Linear(output_dim, output_dim)
        elif self.highway_network == 'Wg(x,y)':
            self.dropg1 = nn.Dropout(p=drop)
            self.dropg2 = nn.Dropout(p=drop)
            self.wg = nn.Linear(2*output_dim, output_dim)
        elif self.highway_network == 'Wg(SteGluFfn(x))':
            self.wg = nn.Linear(output_dim, output_dim)
            self.ste = SteGluFfn(input_dim=input_dim,
                                 hidden_dim=input_dim,
                                 output_dim=output_dim,
                                 drop=0.5,
                                 ste_layers_cnt=1)
        elif self.highway_network == 'Wg(x,y)*y':
            self.drop_wg = nn.Dropout(p=drop)
            self.drop_out = nn.Dropout(p=0.2)
            self.wg = nn.Linear(2*output_dim, output_dim)
        elif self.highway_network == 'Ste(x,y)*y':
            self.drop_wg = nn.Dropout(p=drop)
            self.ste = SteGluFfn(input_dim=2*output_dim,
                                 hidden_dim=input_dim,
                                 output_dim=output_dim,
                                 drop=0.5,
                                 ste_layers_cnt=1)

    def forward(self, x):
        # dense layer 1
        if self.drop_pos == 'input':
            x1_in = self.drop1(x)
        else:
            x1_in = x
        x1 = self.fc1(x1_in)
        if self.norm:
            x1 = self.norm1(x1)
        x1 = self.act1(x1)
        if self.drop_pos == 'output':
            x1 = self.drop1(x1)

        # dense layer 2
        if self.drop_pos == 'input':
            x2_in = self.drop2(torch.cat((x, x1), dim=1))
        else:
            x2_in = torch.cat((x, x1), dim=1)
        x2 = self.fc2(x2_in)
        if self.norm:
            x2 = self.norm2(x2)
        x2 = self.act2(x2)
        if self.drop_pos == 'output':
            x2 = self.drop2(x2)

        # dense layer 3
        if self.drop_pos == 'input':
            x3_in = self.drop3(torch.cat((x, x1, x2), dim=1))
        else:
            x3_in = torch.cat((x, x1, x2), dim=1)
        y = self.fc_out(x3_in)

        if self.highway_network == 'Wg(x)':
            g = torch.sigmoid(self.wg(self.dropg2(x)))
            y = g*y + (1-g)*self.dropg1(x)
        elif self.highway_network == 'Wg(x,y)':
            g = torch.sigmoid(self.wg(self.dropg2(torch.cat((x, y), dim=1))))
            y = g*y + (1-g)*self.dropg1(x)
        elif self.highway_network == 'Wg(SteGluFfn(x))':
            g = torch.sigmoid(self.wg(x))
            y = g*y + (1-g)*self.ste(x)
        elif self.highway_network == 'Wg(x,y)*y':
            g = torch.sigmoid(self.wg(self.drop_wg(torch.cat((x, y), dim=1))))
            y = self.drop_out(g)*y
        elif self.highway_network == 'Ste(x,y)*y':
            g = torch.sigmoid(self.ste(self.drop_wg(torch.cat((x, y), dim=1))))
            y = g*y
        if self.cat_inout:
            y = torch.cat((y, x1), dim=1)
        return y

class DenseLayer4l(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim=4096, output_dim=4096,
                 drop=0.0, drop_pos='output'):
        super(DenseLayer4l, self).__init__()
        self.drop_pos = drop_pos  # only output position implemented

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(p=drop)
        self.fc2 = nn.Linear(input_dim+hidden_dim, hidden_dim)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(p=drop)
        self.fc3 = nn.Linear(input_dim+2*hidden_dim, hidden_dim)
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(p=drop)
        self.fc_out = nn.Linear(input_dim+3*hidden_dim, output_dim)

    def forward(self, x):
        # dense layer 1
        x1_in = x
        x1 = self.act1(self.fc1(x1_in))
        x1 = self.drop1(x1)

        # dense layer 2
        x2_in = torch.cat((x, x1), dim=1)
        x2 = self.act2(self.fc2(x2_in))
        x2 = self.drop2(x2)

        # dense layer 3
        x3_in = torch.cat((x, x1, x2), dim=1)
        x3 = self.act3(self.fc3(x3_in))
        x3 = self.drop3(x3)

        # dense layer 4
        x4_in = torch.cat((x, x1, x2, x3), dim=1)
        y = self.fc_out(x4_in)
        return y

class FcLayer(nn.Module):
    def __init__(self, input_dim=4096, output_dim=4096,
                 drop=0.0, act=None):
        super(FcLayer, self).__init__()
        self.act = act

        self.fc = nn.Linear(input_dim, output_dim)
        if self.act == 'relu':
            self.act_out = nn.ReLU()
        elif self.act == 'gelu':
            self.act_out = gelu
        elif self.act == 'mish':
            self.act_out = Mish()
        elif self.act == 'swish':
            self.act_out = Swish()
        elif self.act == 'sharkfin':
            self.act_out = sharkfin
        elif self.act == 'sinrelu':
            self.act_out = sineReLU
        elif self.act == 'nlrelu':
            self.act_out = nl_relu
        self.drop = nn.Dropout(p=drop)

    def forward(self, x):
        x = self.fc(x)
        if self.act:
            x = self.act_out(x)
        x = self.drop(x)
        return x

class SentenceLinker(nn.Module):
    def __init__(self, config):
        super(SentenceLinker, self).__init__()
        self.s2v_dim = config['s2v_dim']
        self.layers = nn.ModuleList()
        if config['sentence_linker']['input_drop'] != 0.0:
            self.layers.append(nn.Dropout(
                p=config['sentence_linker']['input_drop']))

        for layer in config['sentence_linker']['layers']:
            layer_type = list(layer)[0]
            module_config = layer[layer_type]
            if layer_type == 'SteGluFfn':
                module = SteGluFfn(input_dim=module_config['input_dim'],
                                   hidden_dim=module_config['hidden_dim'],
                                   output_dim=module_config['output_dim'],
                                   drop=module_config['ste_drop'],
                                   ste_layers_cnt=module_config['ste_layers_cnt'])
            elif layer_type == 'GluFfn':
                module = GluFfn(input_dim=module_config['input_dim'],
                                hidden_dim=module_config['hidden_dim'],
                                output_dim=module_config['output_dim'])
            elif layer_type == 'SteFfn':
                module = SteFfn(input_dim=module_config['input_dim'],
                                hidden_dim=module_config['hidden_dim'],
                                output_dim=module_config['output_dim'],
                                drop=module_config['ste_drop'],
                                ste_layers_cnt=module_config['ste_layers_cnt'])
            elif layer_type == 'ResFfn':
                module = ResFfn(input_dim=module_config['input_dim'],
                                hidden_dim=module_config['hidden_dim'],
                                output_dim=module_config['output_dim'],
                                drop=module_config['res_drop'])
            elif layer_type == 'ResGlu':
                module = ResGlu(input_dim=module_config['input_dim'],
                                output_dim=module_config['output_dim'],
                                drop=module_config['res_drop'])
            elif layer_type == 'GateOutFfn':
                module = GateOutFfn(input_dim=module_config['input_dim'],
                                    hidden_dim=module_config['hidden_dim'],
                                    output_dim=module_config['output_dim'],
                                    drop=module_config['res_drop'])
            elif layer_type == 'DenseLayer':
                module = DenseLayer(input_dim=module_config['input_dim'],
                                    hidden_dim=module_config['hidden_dim'],
                                    output_dim=module_config['output_dim'],
                                    drop=module_config['hid_drop'],
                                    drop_pos=module_config['drop_pos'],
                                    norm=module_config['norm'],
                                    highway_network=module_config['highway_network'],
                                    cat_inout=module_config['cat_inout'],
                                    hidden_act=module_config['hidden_act'])
            elif layer_type == 'DenseLayer4l':
                module = DenseLayer4l(input_dim=module_config['input_dim'],
                                      hidden_dim=module_config['hidden_dim'],
                                      output_dim=module_config['output_dim'],
                                      drop=module_config['hid_drop'],
                                      drop_pos=module_config['drop_pos'])
            elif layer_type == 'FcLayer':
                module = FcLayer(input_dim=module_config['input_dim'],
                                 output_dim=module_config['output_dim'],
                                 drop=module_config['drop'],
                                 act=module_config['act'])
            self.layers.append(module)

    def forward(self, x):
        for module in self.layers:
            x = module(x)
        return x
