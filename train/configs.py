import copy
import tensorflow as tf


default_config = {
    'batch_size': 128,
    's2v_dim': 4096,
    'name': '',
    'sentence_linker': {
        'input_drop': 0.0,
        'prev_link_hdim': 512,
        'num_gen_links': 2,
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
    configs[i]['training']['lr'] = 5e-5
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
    configs[i]['name'] = 'bL128_4p1sentValid_inDr.2|2xDenseLayeDr.5r_Lr5-51000e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(1, 2):
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
        },
        {'DenseLayer': {
            'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None,
            'highway_network': None, 'cat_inout': False, 'hidden_act': 'relu'}
        }
    ]
    configs[i]['name'] = 'bL128_4p1sentValid_inDr.2|3xDenseLayerDr.5_Lr1e-4_1000e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(2, 3):
    configs[i]['training']['epochs'] = 1000
    configs[i]['training']['lr'] = 2e-4
    configs[i]['sentence_linker']['input_drop'] = 0.2
    configs[i]['sentence_linker']['layers'] = [
        {'DenseLayer': {
            'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.3, 'drop_pos': 'output', 'norm': None,
            'highway_network': None, 'cat_inout': False, 'hidden_act': 'relu'}
        },
        {'DenseLayer': {
            'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.3, 'drop_pos': 'output', 'norm': None,
            'highway_network': None, 'cat_inout': False, 'hidden_act': 'relu'}
        }
    ]
    configs[i]['name'] = 'bL128_4p1sentValid_inDr.2|2xDenseLayerDr.3_Lr2e-4_1000e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(3, 4):
    configs[i]['training']['epochs'] = 1000
    configs[i]['training']['lr'] = 5e-5
    configs[i]['sentence_linker']['input_drop'] = 0.2
    configs[i]['sentence_linker']['num_gen_links'] = 1
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
    configs[i]['name'] = 'bL128_4pRndValid_1LinksPredValidInData_inDr.2|2xDenseLayeDr.5r_Lr5-5_1000e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(4, 5):
    configs[i]['training']['epochs'] = 500
    configs[i]['training']['lr'] = 5e-5
    configs[i]['sentence_linker']['input_drop'] = 0.2
    configs[i]['sentence_linker']['num_gen_links'] = 1
    configs[i]['sentence_linker']['layers'] = [
        {'DenseLayer': {
            'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None,
            'highway_network': None, 'cat_inout': False, 'hidden_act': 'relu'}
        },
        {'DenseLayer': {
            'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None,
            'highway_network': None, 'cat_inout': False, 'hidden_act': 'relu'}
        },
        {'DenseLayer': {
            'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None,
            'highway_network': None, 'cat_inout': False, 'hidden_act': 'relu'}
        },
        {'DenseLayer': {
            'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None,
            'highway_network': None, 'cat_inout': False, 'hidden_act': 'relu'}
        }
    ]
    configs[i]['name'] = 'bL128_4pRndValid_1LinksPredValidInData_inDr.2|4xDenseLayeDr.5r_Lr5-5_500e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(5, 6):
    configs[i]['training']['epochs'] = 200
    configs[i]['training']['lr'] = 5e-5
    configs[i]['sentence_linker']['input_drop'] = 0.2
    configs[i]['sentence_linker']['num_gen_links'] = 1
    configs[i]['sentence_linker']['layers'] = [
        {'DenseLayer': {
            'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 2048, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None,
            'highway_network': None, 'cat_inout': False, 'hidden_act': 'relu'}
        },
        {'DenseLayer': {
            'input_dim': 2048, 'hidden_dim': 2048, 'output_dim': 2048, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None,
            'highway_network': None, 'cat_inout': False, 'hidden_act': 'relu'}
        },
        {'DenseLayer': {
            'input_dim': 2048, 'hidden_dim': 2048, 'output_dim': 2048, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None,
            'highway_network': None, 'cat_inout': False, 'hidden_act': 'relu'}
        },
        {'DenseLayer': {
            'input_dim': 2048, 'hidden_dim': 2048, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None,
            'highway_network': None, 'cat_inout': False, 'hidden_act': 'relu'}
        },
        {'DenseLayer': {
            'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None,
            'highway_network': None, 'cat_inout': False, 'hidden_act': 'relu'}
        }
    ]
    configs[i]['name'] = 'bL128_4pRndValid_1LinksPredValidInData_inDr.2|5xDenseLayeDr.5r(2hLay2048)_Lr5-5_200e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(6, 7):
    configs[i]['training']['epochs'] = 500
    configs[i]['training']['lr'] = 5e-5
    configs[i]['sentence_linker']['input_drop'] = 0.2
    configs[i]['sentence_linker']['num_gen_links'] = 1
    configs[i]['sentence_linker']['layers'] = [
        {'DenseLayer': {
            'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None,
            'highway_network': None, 'cat_inout': False, 'hidden_act': 'relu'}
        },
        {'DenseLayer': {
            'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None,
            'highway_network': None, 'cat_inout': False, 'hidden_act': 'relu'}
        },
        {'DenseLayer': {
            'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None,
            'highway_network': None, 'cat_inout': False, 'hidden_act': 'relu'}
        },
        {'DenseLayer': {
            'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None,
            'highway_network': None, 'cat_inout': False, 'hidden_act': 'relu'}
        }
    ]
    configs[i]['name'] = 'bL128_4pRndValid_inDr.2|4xDenseLayeDr.5r_Lr5-5_500e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(7, 8):
    configs[i]['training']['epochs'] = 500
    configs[i]['training']['lr'] = 5e-5
    configs[i]['sentence_linker']['input_drop'] = 0.2
    configs[i]['sentence_linker']['num_gen_links'] = 1
    configs[i]['sentence_linker']['layers'] = [
        {'DenseLayer': {
            'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None,
            'highway_network': None, 'cat_inout': False, 'hidden_act': 'relu'}
        },
        {'DenseLayer': {
            'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None,
            'highway_network': None, 'cat_inout': False, 'hidden_act': 'relu'}
        },
        {'DenseLayer': {
            'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None,
            'highway_network': None, 'cat_inout': False, 'hidden_act': 'relu'}
        },
        {'DenseLayer': {
            'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None,
            'highway_network': None, 'cat_inout': False, 'hidden_act': 'relu'}
        }
    ]
    configs[i]['name'] = 'bL128_4pRndValid_trSmoothL_inDr.2|4xDenseLayeDr.5r_Lr5-5_500e_'+str(i)

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