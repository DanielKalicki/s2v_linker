import copy
import tensorflow as tf


default_config = {
    'batch_size': 128,
    's2v_dim': 4096,
    'name': '',
    'sentence_linker': {
        'input_drop': 0.0,
        'layers': [
            # {
            #     'SteGluFfn': {
            #         'input_dim': 4096,
            #         'hidden_dim': 4096,
            #         'output_dim': 4096,
            #         'ste_drop': 0.5
            #     }
            # }
        ]
    },
    'training': {
        'optimizer': 'Adam',
        'clipnorm': None,
        'batch_repeat': 1,
        'lr': 1e-4,
        'lr_discriminator': 2e-4,
        'epochs': 100,
        'log': True,
        'l1_loss': 0.0,
        'l2_loss': 0.0
    }
}

configs = []
for i in range(1000):
    configs.append(copy.deepcopy(default_config))

# -----------------------------------------------------------------------------
for i in range(0, 1):
    configs[i]['training']['epochs'] = 1000
    configs[i]['sentence_linker']['input_drop'] = 0.2
    configs[i]['sentence_linker']['layers'] = [
        {'DenseLayer': {
            'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None,
            'highway_network': None, 'cat_inout': False, 'hidden_act': 'relu'}
        },
        {'DenseLayer': {
            'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None,
            'highway_network': None, 'cat_inout': False, 'hidden_act': 'relu'}
        }
    ]
    configs[i]['name'] = 'bL128_inDr.2|2xDenseLayer_Lr1e-4_1000e_'+str(i)
#
# TODO
# [X] res_drop = [0.1, 0.3, 0.5]
# [X] 3xSteGluFfn
# [ ] gradient drop
# [ ] dropconnect
# [ ] NAC & NALU
# [X] gated ResFfn -> GateOutFfn
# [X] initial drop
# [X] l1, l2 regularization
# [ ] optimizers