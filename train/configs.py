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
        'rnn_model': True,
        'state_vect': False,
        'layers': []
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
    configs[i]['name'] = 'bL128_4p1sentValid_inDr.2|2xDenseLayeDr.5r_Lr5-5_1000e_'+str(i)
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
    configs[i]['name'] = 'bL128_4pRndValid_DiscrFfh2hLr2e-4_inDr.2|4xDenseLayeDr.5_Lr5-5_500e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(8, 9):
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
        }
    ]
    configs[i]['name'] = 'bL128_4pRndValid_inDr.2|3xDenseLayeDr.5r_Lr5-5_500e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(9, 10):
    configs[i]['training']['epochs'] = 500
    configs[i]['training']['lr'] = 5e-6
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
        }
    ]
    configs[i]['name'] = 'bL128_4pRndValid_inDr.2|3xDenseLayeDr.5r_Lr5e-6_500e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(10, 11):
    configs[i]['training']['epochs'] = 500
    configs[i]['training']['lr'] = 5e-6
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
        }
    ]
    configs[i]['name'] = 'bL128_4pRndValid_inDr.2|3xDenseLayeDr.5r_Lr5e-6_500e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(11, 12):
    configs[i]['training']['epochs'] = 500
    configs[i]['training']['lr'] = 5e-6
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
        }
    ]
    configs[i]['name'] = 'bL128_4pRndValid_inDr.2|3xDenseLayeDr.5r_Lr5e-6_500e_restore_log2TrGen_'+str(i)
# -----------------------------------------------------------------------------
for i in range(12, 13):
    configs[i]['training']['epochs'] = 500
    configs[i]['training']['lr'] = 5e-5
    configs[i]['sentence_linker']['input_drop'] = 0.5
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
        }
    ]
    configs[i]['name'] = 'bL128_4pRndValid_inDr.5|3xDenseLayeDr.5r_Lr5-5_500e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(13, 14):
    configs[i]['training']['epochs'] = 500
    configs[i]['training']['lr'] = 5e-5
    configs[i]['sentence_linker']['input_drop'] = 0.0
    configs[i]['sentence_linker']['num_gen_links'] = 1
    configs[i]['sentence_linker']['layers'] = [
        {'DenseLayer': {
            'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.3, 'drop_pos': 'output', 'norm': None,
            'highway_network': None, 'cat_inout': False, 'hidden_act': 'relu'}
        },
        {'DenseLayer': {
            'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.3, 'drop_pos': 'output', 'norm': None,
            'highway_network': None, 'cat_inout': False, 'hidden_act': 'relu'}
        },
        {'DenseLayer': {
            'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.3, 'drop_pos': 'output', 'norm': None,
            'highway_network': None, 'cat_inout': False, 'hidden_act': 'relu'}
        }
    ]
    configs[i]['name'] = 'bL128_4pRndValid_rndSentNoise.3_inDr.0|3xDenseLayeDr.3r_Lr5-5_500e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(14, 15):
    configs[i]['training']['epochs'] = 500
    configs[i]['training']['lr'] = 5e-5
    configs[i]['sentence_linker']['input_drop'] = 0.0
    configs[i]['sentence_linker']['num_gen_links'] = 1
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
    configs[i]['name'] = 'bL128_4pRndValid_rndSentNoise.3_inDr.0|2xDenseLayeDr.3r_Lr5-5_500e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(15, 16):
    configs[i]['training']['epochs'] = 500
    configs[i]['training']['lr'] = 5e-5
    configs[i]['sentence_linker']['input_drop'] = 0.0
    configs[i]['sentence_linker']['num_gen_links'] = 1
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
    configs[i]['name'] = 'bL128_4pRndValid_rndSentDimSwap.3_inDr.0|2xDenseLayeDr.3r_Lr5-5_500e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(16, 17):
    configs[i]['training']['epochs'] = 500
    configs[i]['training']['lr'] = 5e-5
    configs[i]['sentence_linker']['num_gen_links'] = 1
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
    configs[i]['name'] = 'bL128_4pRndValid_rndSentDimSwap.3SenNoise.3GNoise0.1_2xDenseLayeDr.3r_Lr5-5_500e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(17, 18):
    configs[i]['training']['epochs'] = 500
    configs[i]['training']['lr'] = 5e-5
    configs[i]['sentence_linker']['num_gen_links'] = 1
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
    configs[i]['name'] = 'bL128_InpLab_rndSentDimSwap.3SenNoise.3GNoise0.1_2xDenseLayeDr.3r_Lr5-5_500e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(18, 19):
    configs[i]['training']['epochs'] = 500
    configs[i]['training']['lr'] = 5e-5
    configs[i]['sentence_linker']['input_drop'] = 0.0
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
        }
    ]
    configs[i]['name'] = 'bL128_4pRndValid_inDr.0|3xDenseLayeDr.5r_Lr5-5_500e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(19, 20):
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
        }
    ]
    configs[i]['name'] = 'bL128_InpLab_rndSentNoise.3_inDr.2|2xDenseLayeDr.5r_Lr5-5_500e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(20, 21):
    configs[i]['training']['epochs'] = 500
    configs[i]['training']['lr'] = 5e-5
    configs[i]['sentence_linker']['input_drop'] = 0.0
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
    configs[i]['name'] = 'bL128_InpLab_rest1to1_gNoise.1_inDr.0|2xDenseLayeDr.5r_Lr5-5_500e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(21, 22):
    configs[i]['training']['epochs'] = 500
    configs[i]['training']['lr'] = 5e-5
    configs[i]['sentence_linker']['input_drop'] = 0.0
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
    configs[i]['name'] = 'bL128_InpLab_rest1to1_rndSentNoise.1_inDr.0|2xDenseLayeDr.5r_Lr5-5_500e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(22, 23):
    configs[i]['training']['epochs'] = 500
    configs[i]['training']['lr'] = 5e-5
    configs[i]['sentence_linker']['input_drop'] = 0.0
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
    configs[i]['name'] = 'bL128_InpLab_rest1to1_rndDimSentSwap.2_inDr.0|2xDenseLayeDr.5r_Lr5-5_500e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(23, 24):
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
        }
    ]
    configs[i]['name'] = 'bL128_1s_inDr.2|2xDenseLayeDr.5r_Lr5-5_500e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(24, 25):
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
        }
    ]
    configs[i]['name'] = 'bL128_1s_lossThrLog10.0025=0_restore_inDr.2|2xDenseLayeDr.5r_Lr5-5_500e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(25, 26):
    configs[i]['training']['epochs'] = 500
    configs[i]['training']['lr'] = 5e-5
    configs[i]['sentence_linker']['input_drop'] = 0.2
    configs[i]['sentence_linker']['num_gen_links'] = 3
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
    configs[i]['name'] = 'bL128_1s_fixPos3LinksDataPrevLink_inDr.2|2xDenseLayeDr.5r_Lr5-5_500e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(26, 27):
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
            'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096*3, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None,
            'highway_network': None, 'cat_inout': False, 'hidden_act': 'relu'}
        }
    ]
    configs[i]['name'] = 'bL128_1s_fixPos3LinksDataPrevLinkOutx3v3_inDr.2|2xDenseLayeDr.5r_Lr5-5_500e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(27, 28):
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
            'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096*3, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None,
            'highway_network': None, 'cat_inout': False, 'hidden_act': 'relu'}
        }
    ]
    configs[i]['name'] = 'bL128_1s_3LinksDataPrevLinkOutx3v3_inDr.2|2xDenseLayeDr.5r_Lr5-5_500e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(28, 29):
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
            'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096*3, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None,
            'highway_network': None, 'cat_inout': False, 'hidden_act': 'relu'}
        }
    ]
    configs[i]['name'] = 'bL128_1s_3LinksDataPrevLinkOutx3v3_inDr.2|2xDenseLayeDr.5r_Lr5-5_500e_restore_'+str(i)
# -----------------------------------------------------------------------------
for i in range(29, 30):
    configs[i]['training']['epochs'] = 500
    configs[i]['training']['lr'] = 5e-6
    configs[i]['sentence_linker']['input_drop'] = 0.2
    configs[i]['sentence_linker']['num_gen_links'] = 1
    configs[i]['sentence_linker']['layers'] = [
        {'DenseLayer': {
            'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None,
            'highway_network': None, 'cat_inout': False, 'hidden_act': 'relu'}
        },
        {'DenseLayer': {
            'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096*3, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None,
            'highway_network': None, 'cat_inout': False, 'hidden_act': 'relu'}
        }
    ]
    configs[i]['name'] = 'bL128_1s_3LinksDataPrevLinkOutx3v3_inDr.2|2xDenseLayeDr.5r_Lr5-6_500e_restore_'+str(i)
# -----------------------------------------------------------------------------
for i in range(30, 31):
    configs[i]['training']['epochs'] = 500
    configs[i]['training']['lr'] = 5e-7
    configs[i]['sentence_linker']['input_drop'] = 0.01
    configs[i]['sentence_linker']['num_gen_links'] = 1
    configs[i]['sentence_linker']['layers'] = [
        {'DenseLayer': {
            'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.0, 'drop_pos': 'output', 'norm': None,
            'highway_network': None, 'cat_inout': False, 'hidden_act': 'relu'}
        },
        {'DenseLayer': {
            'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096*3, 'hid_drop': 0.0, 'drop_pos': 'output', 'norm': None,
            'highway_network': None, 'cat_inout': False, 'hidden_act': 'relu'}
        }
    ]
    configs[i]['name'] = 'bL128_1s_3LinksDataPrevLinkOutx3v3_inDr.01|2xDenseLayeDr.0r_Lr5-6_500e_restore_'+str(i)
# -----------------------------------------------------------------------------
for i in range(31, 32):
    configs[i]['training']['epochs'] = 500
    configs[i]['training']['lr'] = 5e-5
    configs[i]['sentence_linker']['input_drop'] = 0.2
    configs[i]['sentence_linker']['num_gen_links'] = 3
    configs[i]['sentence_linker']['prev_link_hdim'] = 2048
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
    configs[i]['name'] = 'bL128_1s_3Links_RNNprv2048_inDr.2|2xDenseLayeDr.5r_Lr5-5_500e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(32, 33):
    configs[i]['training']['epochs'] = 500
    configs[i]['training']['lr'] = 5e-6
    configs[i]['sentence_linker']['rnn_model'] = True
    configs[i]['sentence_linker']['input_drop'] = 0.2
    configs[i]['sentence_linker']['num_gen_links'] = 3
    configs[i]['sentence_linker']['prev_link_hdim'] = 2048
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
    configs[i]['name'] = 'bL128_1s_3Links_RNNprv2048_inDr.2|2xDenseLayeDr.5r_Lr5-6_500e_restore_'+str(i)
# -----------------------------------------------------------------------------
for i in range(33, 34):
    configs[i]['training']['epochs'] = 500
    configs[i]['training']['lr'] = 5e-5
    configs[i]['sentence_linker']['rnn_model'] = True
    configs[i]['sentence_linker']['state_vect'] = True
    configs[i]['sentence_linker']['input_drop'] = 0.2
    configs[i]['sentence_linker']['num_gen_links'] = 3
    configs[i]['sentence_linker']['prev_link_hdim'] = 2048
    configs[i]['sentence_linker']['layers'] = [
        {'DenseLayer': {
            'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None,
            'highway_network': None, 'cat_inout': False, 'hidden_act': 'relu'}
        },
        {'DenseLayer': {
            'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096+2048, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None,
            'highway_network': None, 'cat_inout': False, 'hidden_act': 'relu'}
        }
    ]
    configs[i]['name'] = 'bL128_1s_3Links_RNNstate2048_inDr.2|2xDenseLayeDr.5r_Lr5-5_500e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(34, 35):
    configs[i]['training']['epochs'] = 500
    configs[i]['training']['lr'] = 5e-6
    configs[i]['sentence_linker']['rnn_model'] = True
    configs[i]['sentence_linker']['state_vect'] = True
    configs[i]['sentence_linker']['input_drop'] = 0.2
    configs[i]['sentence_linker']['num_gen_links'] = 3
    configs[i]['sentence_linker']['prev_link_hdim'] = 2048
    configs[i]['sentence_linker']['layers'] = [
        {'DenseLayer': {
            'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None,
            'highway_network': None, 'cat_inout': False, 'hidden_act': 'relu'}
        },
        {'DenseLayer': {
            'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096+2048, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None,
            'highway_network': None, 'cat_inout': False, 'hidden_act': 'relu'}
        }
    ]
    configs[i]['name'] = 'bL128_1s_3Links_RNNstate2048_inDr.2|2xDenseLayeDr.5r_Lr5-6_restore_500e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(35, 36):
    configs[i]['training']['epochs'] = 500
    configs[i]['training']['lr'] = 5e-6
    configs[i]['sentence_linker']['rnn_model'] = True
    configs[i]['sentence_linker']['state_vect'] = True
    configs[i]['sentence_linker']['input_drop'] = 0.01
    configs[i]['sentence_linker']['num_gen_links'] = 3
    configs[i]['sentence_linker']['prev_link_hdim'] = 2048
    configs[i]['sentence_linker']['layers'] = [
        {'DenseLayer': {
            'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.0, 'drop_pos': 'output', 'norm': None,
            'highway_network': None, 'cat_inout': False, 'hidden_act': 'relu'}
        },
        {'DenseLayer': {
            'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096+2048, 'hid_drop': 0.0, 'drop_pos': 'output', 'norm': None,
            'highway_network': None, 'cat_inout': False, 'hidden_act': 'relu'}
        }
    ]
    configs[i]['name'] = 'bL128_1s_3Links_RNNstate2048_inDr.01|2xDenseLayeDr.0r_Lr5-6_restore_500e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(36, 37):
    configs[i]['training']['epochs'] = 500
    configs[i]['training']['lr'] = 5e-6
    configs[i]['sentence_linker']['rnn_model'] = True
    configs[i]['sentence_linker']['state_vect'] = True
    configs[i]['sentence_linker']['input_drop'] = 0.01
    configs[i]['sentence_linker']['num_gen_links'] = 4
    configs[i]['sentence_linker']['prev_link_hdim'] = 2048
    configs[i]['sentence_linker']['layers'] = [
        {'DenseLayer': {
            'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.0, 'drop_pos': 'output', 'norm': None,
            'highway_network': None, 'cat_inout': False, 'hidden_act': 'relu'}
        },
        {'DenseLayer': {
            'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096+2048, 'hid_drop': 0.0, 'drop_pos': 'output', 'norm': None,
            'highway_network': None, 'cat_inout': False, 'hidden_act': 'relu'}
        }
    ]
    configs[i]['name'] = 'bL128_1s_4Links_RNNstate2048_inDr.01|2xDenseLayeDr.0r_Lr5-6_restore_500e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(37, 38):
    configs[i]['training']['epochs'] = 500
    configs[i]['training']['lr'] = 5e-6
    configs[i]['sentence_linker']['rnn_model'] = True
    configs[i]['sentence_linker']['state_vect'] = True
    configs[i]['sentence_linker']['input_drop'] = 0.01
    configs[i]['sentence_linker']['num_gen_links'] = 5
    configs[i]['sentence_linker']['prev_link_hdim'] = 2048
    configs[i]['sentence_linker']['layers'] = [
        {'DenseLayer': {
            'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.0, 'drop_pos': 'output', 'norm': None,
            'highway_network': None, 'cat_inout': False, 'hidden_act': 'relu'}
        },
        {'DenseLayer': {
            'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096+2048, 'hid_drop': 0.0, 'drop_pos': 'output', 'norm': None,
            'highway_network': None, 'cat_inout': False, 'hidden_act': 'relu'}
        }
    ]
    configs[i]['name'] = 'bL128_1s_5Links_RNNstate2048_inDr.01|2xDenseLayeDr.0r_Lr5-6_restore_500e_'+str(i)
