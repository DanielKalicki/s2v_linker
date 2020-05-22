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
for i in range(0, 4):
    configs[i]['sentence_linker']['layers'] = [
        {
            'SteGluFfn': {
                'input_dim': 4096,
                'hidden_dim': 4096,
                'output_dim': 4096,
                'ste_drop': 0.5
            }
        }
    ]
    configs[i]['name'] = 'bL128_SteGluFfnx1_Lr1e-4_'+str(i)

# -----------------------------------------------------------------------------
for i in range(4, 8):
    configs[i]['sentence_linker']['layers'] = [
        {
            'SteGluFfn': {
                'input_dim': 4096,
                'hidden_dim': 2*4096,
                'output_dim': 4096,
                'ste_drop': 0.5
            }
        }
    ]
    configs[i]['name'] = 'bL128_SteGluFfnx2_Lr1e-4_'+str(i)

# -----------------------------------------------------------------------------
for i in range(8, 12):
    configs[i]['sentence_linker']['layers'] = [
        {
            'SteGluFfn': {
                'input_dim': 4096,
                'hidden_dim': 4*4096,
                'output_dim': 4096,
                'ste_drop': 0.5
            }
        }
    ]
    configs[i]['name'] = 'bL128_SteGluFfnx4_Lr1e-4_'+str(i)

# -----------------------------------------------------------------------------
for i in range(12, 16):
    configs[i]['sentence_linker']['layers'] = [
        {
            'SteGluFfn': {
                'input_dim': 4096,
                'hidden_dim': int(4096/2),
                'output_dim': 4096,
                'ste_drop': 0.5
            }
        }
    ]
    configs[i]['name'] = 'bL128_SteGluFfnd2_Lr1e-4_'+str(i)

# -----------------------------------------------------------------------------
for i in range(16, 20):
    configs[i]['sentence_linker']['layers'] = [
        {
            'SteGluFfn': {
                'input_dim': 4096,
                'hidden_dim': 4096,
                'output_dim': 4096,
                'ste_drop': 0.3
            }
        }
    ]
    configs[i]['name'] = 'bL128_SteGluFfnx1Dr.3_Lr1e-4_'+str(i)

# -----------------------------------------------------------------------------
for i in range(20, 24):
    configs[i]['sentence_linker']['layers'] = [
        {
            'GluFfn': {
                'input_dim': 4096,
                'hidden_dim': 4096,
                'output_dim': 4096
            }
        }
    ]
    configs[i]['name'] = 'bL128_GluFfnx1_Lr1e-4_'+str(i)

# -----------------------------------------------------------------------------
for i in range(24, 28):
    configs[i]['sentence_linker']['layers'] = [
        {
            'GluFfn': {
                'input_dim': 4096,
                'hidden_dim': 2*4096,
                'output_dim': 4096
            }
        }
    ]
    configs[i]['name'] = 'bL128_GluFfnx2_Lr1e-4_'+str(i)

# -----------------------------------------------------------------------------
for i in range(28, 32):
    configs[i]['sentence_linker']['layers'] = [
        {
            'SteGluFfn': {
                'input_dim': 4096,
                'hidden_dim': 4096,
                'output_dim': 4096,
                'ste_drop': 0.5
            }
        },
        {
            'SteGluFfn': {
                'input_dim': 4096,
                'hidden_dim': 4096,
                'output_dim': 4096,
                'ste_drop': 0.5
            }
        }
    ]
    configs[i]['name'] = 'bL128_2xSteGluFfnx1_Lr1e-4_'+str(i)

# -----------------------------------------------------------------------------
for i in range(32, 36):
    configs[i]['sentence_linker']['layers'] = [
        {
            'GluFfn': {
                'input_dim': 4096,
                'hidden_dim': 4096,
                'output_dim': 4096
            }
        },
        {
            'GluFfn': {
                'input_dim': 4096,
                'hidden_dim': 4096,
                'output_dim': 4096
            }
        }
    ]
    configs[i]['name'] = 'bL128_2xGluFfnx1_Lr1e-4_'+str(i)

# -----------------------------------------------------------------------------
for i in range(36, 40):
    configs[i]['sentence_linker']['layers'] = [
        {
            'SteGluFfn': {
                'input_dim': 4096,
                'hidden_dim': 4096,
                'output_dim': 4096,
                'ste_drop': 0.5
            }
        },
        {
            'SteGluFfn': {
                'input_dim': 4096,
                'hidden_dim': 4096,
                'output_dim': 4096,
                'ste_drop': 0.5
            }
        },
        {
            'SteGluFfn': {
                'input_dim': 4096,
                'hidden_dim': 4096,
                'output_dim': 4096,
                'ste_drop': 0.5
            }
        }
    ]
    configs[i]['name'] = 'bL128_3xSteGluFfnx1_Lr1e-4_'+str(i)

# -----------------------------------------------------------------------------
for i in range(40, 44):
    configs[i]['sentence_linker']['layers'] = [
        {
            'SteFfn': {
                'input_dim': 4096,
                'hidden_dim': 4096,
                'output_dim': 4096,
                'ste_drop': 0.5
            }
        }
    ]
    configs[i]['name'] = 'bL128_SteFfnx1_Lr1e-4_'+str(i)
# -----------------------------------------------------------------------------
for i in range(44, 48):
    configs[i]['sentence_linker']['layers'] = [
        {
            'SteFfn': {
                'input_dim': 4096,
                'hidden_dim': 2*4096,
                'output_dim': 4096,
                'ste_drop': 0.5
            }
        }
    ]
    configs[i]['name'] = 'bL128_SteFfnx2_Lr1e-4_'+str(i)
# -----------------------------------------------------------------------------
for i in range(48, 52):
    configs[i]['sentence_linker']['layers'] = [
        {
            'SteFfn': {
                'input_dim': 4096,
                'hidden_dim': 4096,
                'output_dim': 4096,
                'ste_drop': 0.5
            }
        },
        {
            'SteFfn': {
                'input_dim': 4096,
                'hidden_dim': 4096,
                'output_dim': 4096,
                'ste_drop': 0.5
            }
        }
    ]
    configs[i]['name'] = 'bL128_2xSteFfnx1_Lr1e-4_'+str(i)
# -----------------------------------------------------------------------------
for i in range(52, 55):
    configs[i]['sentence_linker']['layers'] = [
        {
            'SteGluFfn': {
                'input_dim': 4096,
                'hidden_dim': 4096,
                'output_dim': 4096,
                'ste_drop': 0.5
            }
        },
        {
            'ResFfn': {
                'input_dim': 4096,
                'hidden_dim': 4096,
                'output_dim': 4096,
                'res_drop': 0.0
            }
        }
    ]
    configs[i]['name'] = 'bL128_SteGluFfnx1|ResFfnx1_Lr1e-4_'+str(i)
# -----------------------------------------------------------------------------
for i in range(55, 58):
    configs[i]['sentence_linker']['layers'] = [
        {
            'SteGluFfn': {
                'input_dim': 4096,
                'hidden_dim': 4096,
                'output_dim': 4096,
                'ste_drop': 0.5
            }
        },
        {
            'SteGluFfn': {
                'input_dim': 4096,
                'hidden_dim': 4096,
                'output_dim': 4096,
                'ste_drop': 0.5
            }
        },
        {
            'ResFfn': {
                'input_dim': 4096,
                'hidden_dim': 4096,
                'output_dim': 4096,
                'res_drop': 0.0
            }
        }
    ]
    configs[i]['name'] = 'bL128_2xSteGluFfnx1|ResFfnx1_Lr1e-4_'+str(i)
# -----------------------------------------------------------------------------
for i in range(58, 61):
    configs[i]['sentence_linker']['layers'] = [
        {
            'SteGluFfn': {
                'input_dim': 4096,
                'hidden_dim': 4096,
                'output_dim': 4096,
                'ste_drop': 0.5
            }
        },
        {
            'ResGlu': {
                'input_dim': 4096,
                'output_dim': 4096,
                'res_drop': 0.0
            }
        }
    ]
    configs[i]['name'] = 'bL128_SteGluFfnx1|ResGlu_Lr1e-4_'+str(i)
# -----------------------------------------------------------------------------
for i in range(61, 64):
    configs[i]['sentence_linker']['layers'] = [
        {
            'SteGluFfn': {
                'input_dim': 4096,
                'hidden_dim': 4096,
                'output_dim': 4096,
                'ste_drop': 0.5
            }
        },
        {
            'SteGluFfn': {
                'input_dim': 4096,
                'hidden_dim': 4096,
                'output_dim': 4096,
                'ste_drop': 0.5
            }
        },
        {
            'ResGlu': {
                'input_dim': 4096,
                'output_dim': 4096,
                'res_drop': 0.0
            }
        }
    ]
    configs[i]['name'] = 'bL128_2xSteGluFfnx1|ResGlu_Lr1e-4_'+str(i)
# -----------------------------------------------------------------------------
for i in range(64, 67):
    configs[i]['sentence_linker']['layers'] = [
        {
            'SteGluFfn': {
                'input_dim': 4096,
                'hidden_dim': 4096,
                'output_dim': 4096,
                'ste_drop': 0.5
            }
        },
        {
            'ResFfn': {
                'input_dim': 4096,
                'hidden_dim': 4096,
                'output_dim': 4096,
                'res_drop': 0.0
            }
        },
        {
            'ResFfn': {
                'input_dim': 4096,
                'hidden_dim': 4096,
                'output_dim': 4096,
                'res_drop': 0.0
            }
        }
    ]
    configs[i]['name'] = 'bL128_SteGluFfnx1|2xResFfnx1_Lr1e-4_'+str(i)
# -----------------------------------------------------------------------------
for i in range(67, 70):
    configs[i]['sentence_linker']['layers'] = [
        {
            'SteGluFfn': {
                'input_dim': 4096,
                'hidden_dim': 4096,
                'output_dim': 4096,
                'ste_drop': 0.5
            }
        },
        {
            'ResGlu': {
                'input_dim': 4096,
                'output_dim': 4096,
                'res_drop': 0.0
            }
        },
        {
            'ResGlu': {
                'input_dim': 4096,
                'output_dim': 4096,
                'res_drop': 0.0
            }
        }
    ]
    configs[i]['name'] = 'bL128_SteGluFfnx1|2xResGlu_Lr1e-4_'+str(i)
# -----------------------------------------------------------------------------
for i in range(70, 73):
    configs[i]['sentence_linker']['layers'] = [
        {
            'SteGluFfn': {
                'input_dim': 4096,
                'hidden_dim': 4096,
                'output_dim': 4096,
                'ste_drop': 0.5
            }
        },
        {
            'GateOutFfn': {
                'input_dim': 4096,
                'hidden_dim': 4096,
                'output_dim': 4096,
                'res_drop': 0.0
            }
        }
    ]
    configs[i]['name'] = 'bL128_SteGluFfnx1|GateOutFfnx1_Lr1e-4_'+str(i)
# -----------------------------------------------------------------------------
for i in range(73, 76):
    configs[i]['sentence_linker']['layers'] = [
        {
            'SteGluFfn': {
                'input_dim': 4096,
                'hidden_dim': 4096,
                'output_dim': 4096,
                'ste_drop': 0.5
            }
        },
        {
            'GateOutFfn': {
                'input_dim': 4096,
                'hidden_dim': 4096,
                'output_dim': 4096,
                'res_drop': 0.0
            }
        },
        {
            'GateOutFfn': {
                'input_dim': 4096,
                'hidden_dim': 4096,
                'output_dim': 4096,
                'res_drop': 0.0
            }
        }
    ]
    configs[i]['name'] = 'bL128_SteGluFfnx1|2xGateOutFfnx1_Lr1e-4_'+str(i)
# -----------------------------------------------------------------------------
for i in range(76, 79):
    configs[i]['sentence_linker']['layers'] = [
        {
            'SteGluFfn': {
                'input_dim': 4096,
                'hidden_dim': 4096,
                'output_dim': 4096,
                'ste_drop': 0.5
            }
        },
        {
            'ResFfn': {
                'input_dim': 4096,
                'hidden_dim': 4096,
                'output_dim': 4096,
                'res_drop': 0.3
            }
        },
        {
            'ResFfn': {
                'input_dim': 4096,
                'hidden_dim': 4096,
                'output_dim': 4096,
                'res_drop': 0.3
            }
        }
    ]
    configs[i]['name'] = 'bL128_SteGluFfnx1|2xResDr.3Ffnx1_Lr1e-4_'+str(i)
# -----------------------------------------------------------------------------
for i in range(79, 82):
    configs[i]['sentence_linker']['layers'] = [
        {
            'SteGluFfn': {
                'input_dim': 4096,
                'hidden_dim': 4096,
                'output_dim': 4096,
                'ste_drop': 0.5
            }
        },
        {
            'ResFfn': {
                'input_dim': 4096,
                'hidden_dim': 4096,
                'output_dim': 4096,
                'res_drop': 0.1
            }
        },
        {
            'ResFfn': {
                'input_dim': 4096,
                'hidden_dim': 4096,
                'output_dim': 4096,
                'res_drop': 0.1
            }
        }
    ]
    configs[i]['name'] = 'bL128_SteGluFfnx1|2xResDr.1Ffnx1_Lr1e-4_'+str(i)
# -----------------------------------------------------------------------------
for i in range(82, 85):
    configs[i]['sentence_linker']['layers'] = [
        {
            'SteGluFfn': {
                'input_dim': 4096,
                'hidden_dim': 4096,
                'output_dim': 4096,
                'ste_drop': 0.5
            }
        },
        {
            'SteGluFfn': {
                'input_dim': 4096,
                'hidden_dim': 4096,
                'output_dim': 4096,
                'ste_drop': 0.5
            }
        },
        {
            'ResFfn': {
                'input_dim': 4096,
                'hidden_dim': 4096,
                'output_dim': 4096,
                'res_drop': 0.1
            }
        }
    ]
    configs[i]['name'] = 'bL128_2xSteGluFfnx1|ResDr.1Ffnx1_Lr1e-4_'+str(i)
# -----------------------------------------------------------------------------
for i in range(85, 88):
    configs[i]['sentence_linker']['layers'] = [
        {
            'SteGluFfn': {
                'input_dim': 4096,
                'hidden_dim': 4096,
                'output_dim': 4096,
                'ste_drop': 0.5
            }
        },
        {
            'SteGluFfn': {
                'input_dim': 4096,
                'hidden_dim': 4096,
                'output_dim': 4096,
                'ste_drop': 0.5
            }
        },
        {
            'ResFfn': {
                'input_dim': 4096,
                'hidden_dim': 4096,
                'output_dim': 4096,
                'res_drop': 0.3
            }
        }
    ]
    configs[i]['name'] = 'bL128_2xSteGluFfnx1|ResDr.3Ffnx1_Lr1e-4_'+str(i)
# -----------------------------------------------------------------------------
for i in range(88, 91):
    configs[i]['sentence_linker']['input_drop'] = 0.1
    configs[i]['sentence_linker']['layers'] = [
        {
            'SteGluFfn': {
                'input_dim': 4096,
                'hidden_dim': 4096,
                'output_dim': 4096,
                'ste_drop': 0.5
            }
        },
        {
            'SteGluFfn': {
                'input_dim': 4096,
                'hidden_dim': 4096,
                'output_dim': 4096,
                'ste_drop': 0.5
            }
        }
    ]
    configs[i]['name'] = 'bL128_inDr.1|2xSteGluFfnx1_Lr1e-4_'+str(i)
# -----------------------------------------------------------------------------
for i in range(91, 94):
    configs[i]['sentence_linker']['input_drop'] = 0.2
    configs[i]['sentence_linker']['layers'] = [
        {
            'SteGluFfn': {
                'input_dim': 4096,
                'hidden_dim': 4096,
                'output_dim': 4096,
                'ste_drop': 0.5
            }
        },
        {
            'SteGluFfn': {
                'input_dim': 4096,
                'hidden_dim': 4096,
                'output_dim': 4096,
                'ste_drop': 0.5
            }
        }
    ]
    configs[i]['name'] = 'bL128_inDr.2|2xSteGluFfnx1_Lr1e-4_'+str(i)
# -----------------------------------------------------------------------------
for i in range(94, 97):
    configs[i]['sentence_linker']['layers'] = [
        {
            'SteGluFfn': {
                'input_dim': 4096,
                'hidden_dim': 4096,
                'output_dim': 4096,
                'ste_drop': 0.5
            }
        },
        {
            'SteGluFfn': {
                'input_dim': 4096,
                'hidden_dim': 4096,
                'output_dim': 4096,
                'ste_drop': 0.5
            }
        },
        {
            'ResFfn': {
                'input_dim': 4096,
                'hidden_dim': 4096,
                'output_dim': 4096,
                'res_drop': 0.3
            }
        },
        {
            'ResFfn': {
                'input_dim': 4096,
                'hidden_dim': 4096,
                'output_dim': 4096,
                'res_drop': 0.3
            }
        }
    ]
    configs[i]['name'] = 'bL128_2xSteGluFfnx1|2xResDr.3Ffnx1_Lr1e-4_'+str(i)
# -----------------------------------------------------------------------------
for i in range(97, 100):
    configs[i]['sentence_linker']['layers'] = [
        {
            'SteGluFfn': {
                'input_dim': 4096,
                'hidden_dim': 4096,
                'output_dim': 4096,
                'ste_drop': 0.5
            }
        },
        {
            'SteGluFfn': {
                'input_dim': 4096,
                'hidden_dim': 4096,
                'output_dim': 4096,
                'ste_drop': 0.5
            }
        },
        {
            'ResFfn': {
                'input_dim': 4096,
                'hidden_dim': 4096,
                'output_dim': 4096,
                'res_drop': 0.1
            }
        },
        {
            'ResFfn': {
                'input_dim': 4096,
                'hidden_dim': 4096,
                'output_dim': 4096,
                'res_drop': 0.1
            }
        }
    ]
    configs[i]['name'] = 'bL128_2xSteGluFfnx1|2xResDr.1Ffnx1_Lr1e-4_'+str(i)
# -----------------------------------------------------------------------------
for i in range(100, 103):
    configs[i]['sentence_linker']['layers'] = [
        {
            'SteGluFfn': {
                'input_dim': 4096,
                'hidden_dim': int(4096/2),
                'output_dim': 4096,
                'ste_drop': 0.5
            }
        },
        {
            'SteGluFfn': {
                'input_dim': 4096,
                'hidden_dim': int(4096/2),
                'output_dim': 4096,
                'ste_drop': 0.5
            }
        }
    ]
    configs[i]['name'] = 'bL128_2xSteGluFfnd2_Lr1e-4_'+str(i)
# -----------------------------------------------------------------------------
for i in range(103, 106):
    configs[i]['sentence_linker']['layers'] = [
        {
            'SteGluFfn': {
                'input_dim': 4096,
                'hidden_dim': int(4096/2),
                'output_dim': 4096,
                'ste_drop': 0.5
            }
        },
        {
            'SteGluFfn': {
                'input_dim': 4096,
                'hidden_dim': int(4096/2),
                'output_dim': 4096,
                'ste_drop': 0.5
            }
        },
        {
            'SteGluFfn': {
                'input_dim': 4096,
                'hidden_dim': int(4096/2),
                'output_dim': 4096,
                'ste_drop': 0.5
            }
        }
    ]
    configs[i]['name'] = 'bL128_3xSteGluFfnd2_Lr1e-4_'+str(i)
# -----------------------------------------------------------------------------
for i in range(106, 109):
    configs[i]['sentence_linker']['layers'] = [
        {
            'SteGluFfn': {
                'input_dim': 4096,
                'hidden_dim': int(4096/4),
                'output_dim': 4096,
                'ste_drop': 0.5
            }
        },
        {
            'SteGluFfn': {
                'input_dim': 4096,
                'hidden_dim': int(4096/4),
                'output_dim': 4096,
                'ste_drop': 0.5
            }
        }
    ]
    configs[i]['name'] = 'bL128_2xSteGluFfnd4_Lr1e-4_'+str(i)
# -----------------------------------------------------------------------------
for i in range(109, 112):
    configs[i]['sentence_linker']['layers'] = [
        {
            'SteGluFfn': {
                'input_dim': 4096,
                'hidden_dim': int(4096/4),
                'output_dim': 4096,
                'ste_drop': 0.5
            }
        },
        {
            'SteGluFfn': {
                'input_dim': 4096,
                'hidden_dim': int(4096/4),
                'output_dim': 4096,
                'ste_drop': 0.5
            }
        },
        {
            'SteGluFfn': {
                'input_dim': 4096,
                'hidden_dim': int(4096/4),
                'output_dim': 4096,
                'ste_drop': 0.5
            }
        }
    ]
    configs[i]['name'] = 'bL128_3xSteGluFfnd4_Lr1e-4_'+str(i)
# -----------------------------------------------------------------------------
for i in range(112, 115):
    configs[i]['sentence_linker']['layers'] = [
        {
            'SteGluFfn': {
                'input_dim': 4096,
                'hidden_dim': int(4096/4),
                'output_dim': 4096,
                'ste_drop': 0.5
            }
        }
    ]
    configs[i]['name'] = 'bL128_SteGluFfnd4_Lr1e-4_'+str(i)
# -----------------------------------------------------------------------------
for i in range(115, 118):
    configs[i]['sentence_linker']['layers'] = [
        {
            'SteGluFfn': {
                'input_dim': 4096,
                'hidden_dim': int(4096/8),
                'output_dim': 4096,
                'ste_drop': 0.5
            }
        }
    ]
    configs[i]['name'] = 'bL128_SteGluFfnd8_Lr1e-4_'+str(i)
# -----------------------------------------------------------------------------
for i in range(118, 121):
    configs[i]['sentence_linker']['input_drop'] = 0.1
    configs[i]['sentence_linker']['layers'] = [
        {
            'SteFfn': {
                'input_dim': 4096,
                'hidden_dim': 4096,
                'output_dim': 4096,
                'ste_drop': 0.5
            }
        },
        {
            'SteFfn': {
                'input_dim': 4096,
                'hidden_dim': 4096,
                'output_dim': 4096,
                'ste_drop': 0.5
            }
        }
    ]
    configs[i]['name'] = 'bL128_inDr.1|2xSteFfnx1_Lr1e-4_'+str(i)
# -----------------------------------------------------------------------------
for i in range(121, 124):
    configs[i]['sentence_linker']['input_drop'] = 0.2
    configs[i]['sentence_linker']['layers'] = [
        {
            'SteFfn': {
                'input_dim': 4096,
                'hidden_dim': 4096,
                'output_dim': 4096,
                'ste_drop': 0.5
            }
        },
        {
            'SteFfn': {
                'input_dim': 4096,
                'hidden_dim': 4096,
                'output_dim': 4096,
                'ste_drop': 0.5
            }
        }
    ]
    configs[i]['name'] = 'bL128_inDr.2|2xSteFfnx1_Lr1e-4_'+str(i)
# -----------------------------------------------------------------------------
for i in range(124, 126):
    configs[i]['training']['epochs'] = 300
    configs[i]['sentence_linker']['layers'] = [
        {
            'SteFfn': {
                'input_dim': 4096,
                'hidden_dim': 4096,
                'output_dim': 4096,
                'ste_drop': 0.5,
                'ste_layers_cnt': 4
            }
        },
        {
            'SteFfn': {
                'input_dim': 4096,
                'hidden_dim': 4096,
                'output_dim': 4096,
                'ste_drop': 0.5,
                'ste_layers_cnt': 4
            }
        }
    ]
    configs[i]['name'] = 'bL128_2xSteFfnx1_Lr1e-4_300e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(126, 128):
    configs[i]['training']['epochs'] = 500
    configs[i]['sentence_linker']['layers'] = [
        {
            'SteGluFfn': {
                'input_dim': 4096,
                'hidden_dim': 4096,
                'output_dim': 4096,
                'ste_drop': 0.5,
                'ste_layers_cnt': 4
            }
        },
        {
            'SteGluFfn': {
                'input_dim': 4096,
                'hidden_dim': 4096,
                'output_dim': 4096,
                'ste_drop': 0.5,
                'ste_layers_cnt': 4
            }
        }
    ]
    configs[i]['name'] = 'bL128_2xSteGluFfnx1_Lr1e-4_500e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(128, 130):
    configs[i]['training']['epochs'] = 300
    configs[i]['sentence_linker']['input_drop'] = 0.2
    configs[i]['sentence_linker']['layers'] = [
        {
            'SteFfn': {
                'input_dim': 4096,
                'hidden_dim': 4096,
                'output_dim': 4096,
                'ste_drop': 0.5,
                'ste_layers_cnt': 4
            }
        },
        {
            'SteFfn': {
                'input_dim': 4096,
                'hidden_dim': 4096,
                'output_dim': 4096,
                'ste_drop': 0.5,
                'ste_layers_cnt': 4
            }
        }
    ]
    configs[i]['name'] = 'bL128_inDr.2|2xSteFfnx1_Lr1e-4_300e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(130, 132):
    configs[i]['training']['epochs'] = 500
    configs[i]['sentence_linker']['input_drop'] = 0.2
    configs[i]['sentence_linker']['layers'] = [
        {
            'SteGluFfn': {
                'input_dim': 4096,
                'hidden_dim': 4096,
                'output_dim': 4096,
                'ste_drop': 0.5,
                'ste_layers_cnt': 4
            }
        },
        {
            'SteGluFfn': {
                'input_dim': 4096,
                'hidden_dim': 4096,
                'output_dim': 4096,
                'ste_drop': 0.5,
                'ste_layers_cnt': 4
            }
        }
    ]
    configs[i]['name'] = 'bL128_inDr.2|2xSteGluFfnx1_Lr1e-4_500e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(132, 135):
    configs[i]['training']['epochs'] = 500
    configs[i]['sentence_linker']['layers'] = [
        {
            'SteGluFfn': {
                'input_dim': 4096,
                'hidden_dim': 4096,
                'output_dim': 4096,
                'ste_drop': 0.5,
                'ste_layers_cnt': 2
            }
        },
        {
            'SteGluFfn': {
                'input_dim': 4096,
                'hidden_dim': 4096,
                'output_dim': 4096,
                'ste_drop': 0.5,
                'ste_layers_cnt': 2
            }
        }
    ]
    configs[i]['name'] = 'bL128_2xSte2GluFfnx1_Lr1e-4_500e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(135, 138):
    configs[i]['training']['epochs'] = 500
    configs[i]['sentence_linker']['layers'] = [
        {
            'SteGluFfn': {
                'input_dim': 4096,
                'hidden_dim': 4096,
                'output_dim': 4096,
                'ste_drop': 0.5,
                'ste_layers_cnt': 1
            }
        },
        {
            'SteGluFfn': {
                'input_dim': 4096,
                'hidden_dim': 4096,
                'output_dim': 4096,
                'ste_drop': 0.5,
                'ste_layers_cnt': 1
            }
        }
    ]
    configs[i]['name'] = 'bL128_2xSte1GluFfnx1_Lr1e-4_500e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(138, 141):
    configs[i]['training']['epochs'] = 500
    configs[i]['sentence_linker']['layers'] = [
        {
            'SteGluFfn': {
                'input_dim': 4096,
                'hidden_dim': 4096,
                'output_dim': 4096,
                'ste_drop': 0.5,
                'ste_layers_cnt': 6
            }
        },
        {
            'SteGluFfn': {
                'input_dim': 4096,
                'hidden_dim': 4096,
                'output_dim': 4096,
                'ste_drop': 0.5,
                'ste_layers_cnt': 6
            }
        }
    ]
    configs[i]['name'] = 'bL128_2xSte6GluFfnx1_Lr1e-4_500e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(141, 144):
    configs[i]['training']['epochs'] = 500
    configs[i]['sentence_linker']['layers'] = [
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.2, 'drop_pos': 'output'
            }
        }
    ]
    configs[i]['name'] = 'bL128_DenseLayerDr.2_Lr1e-4_500e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(144, 147):
    configs[i]['training']['epochs'] = 500
    configs[i]['sentence_linker']['input_drop'] = 0.2
    configs[i]['sentence_linker']['layers'] = [
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output'
            }
        }
    ]
    configs[i]['name'] = 'bL128_inDr.2|DenseLayerDr.5_Lr1e-4_500e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(147, 148):
    configs[i]['training']['epochs'] = 500
    configs[i]['sentence_linker']['input_drop'] = 0.2
    configs[i]['sentence_linker']['layers'] = [
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output'
            }
        },
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output'
            }
        }
    ]
    configs[i]['name'] = 'bL128_inDr.2|2xDenseLayerDr.5_Lr1e-4_500e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(148, 149):
    configs[i]['training']['epochs'] = 500
    configs[i]['sentence_linker']['input_drop'] = 0.2
    configs[i]['sentence_linker']['layers'] = [
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 2048, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output'
            }
        },
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 2048, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output'
            }
        }
    ]
    configs[i]['name'] = 'bL128_inDr.2|2xDenseLayerHid2048Dr.5_Lr1e-4_500e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(149, 150):
    configs[i]['training']['epochs'] = 500
    configs[i]['sentence_linker']['input_drop'] = 0.2
    configs[i]['sentence_linker']['layers'] = [
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 1024, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output'
            }
        },
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 1024, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output'
            }
        }
    ]
    configs[i]['name'] = 'bL128_inDr.2|2xDenseLayerHid1024Dr.5_Lr1e-4_500e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(150, 151):
    configs[i]['training']['epochs'] = 500
    configs[i]['sentence_linker']['layers'] = [
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 1024, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'input'
            }
        },
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 1024, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'input'
            }
        },
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 1024, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'input'
            }
        }
    ]
    configs[i]['name'] = 'bL128_3xDenseLayerHid1024LayInDr.5_Lr1e-4_500e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(151, 152):
    configs[i]['training']['epochs'] = 500
    configs[i]['sentence_linker']['input_drop'] = 0.2
    configs[i]['sentence_linker']['layers'] = [
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 2048, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output'
            }
        },
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 2048, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output'
            }
        },
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 2048, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output'
            }
        }
    ]
    configs[i]['name'] = 'bL128_inDr.2|3xDenseLayerHid2048LayOutDr.5_Lr1e-4_500e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(152, 153):
    configs[i]['training']['epochs'] = 500
    configs[i]['sentence_linker']['input_drop'] = 0.2
    configs[i]['sentence_linker']['layers'] = [
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 1024, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output'
            }
        },
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 1024, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output'
            }
        },
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 1024, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output'
            }
        },
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 1024, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output'
            }
        }
    ]
    configs[i]['name'] = 'bL128_inDr.2|4xDenseLayerHid1024LayOutDr.5_Lr1e-4_500e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(153, 154):
    configs[i]['training']['epochs'] = 500
    configs[i]['sentence_linker']['input_drop'] = 0.2
    configs[i]['sentence_linker']['layers'] = [
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 2048, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output'
            }
        },
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 2048, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output'
            }
        },
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 2048, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output'
            }
        },
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 2048, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output'
            }
        }
    ]
    configs[i]['name'] = 'bL128_inDr.2|4xDenseLayerHid2048LayOutDr.5_Lr1e-4_500e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(154, 155):
    configs[i]['training']['epochs'] = 500
    configs[i]['sentence_linker']['input_drop'] = 0.2
    configs[i]['sentence_linker']['layers'] = [
        {
            'DenseLayer4l': {
                'input_dim': 4096, 'hidden_dim': 1024, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output'
            }
        },
        {
            'DenseLayer4l': {
                'input_dim': 4096, 'hidden_dim': 1024, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output'
            }
        }
    ]
    configs[i]['name'] = 'bL128_inDr.2|2xDenseLayer4lHid1024LayOutDr.5_Lr1e-4_500e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(155, 156):
    configs[i]['training']['epochs'] = 500
    configs[i]['sentence_linker']['input_drop'] = 0.2
    configs[i]['sentence_linker']['layers'] = [
        {
            'SteGluFfn': {
                'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'ste_drop': 0.5, 'ste_layers_cnt': 1
            }
        },
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 1024, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None
            }
        },
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 1024, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None
            }
        }
    ]
    configs[i]['name'] = 'bL128_inDr.2|Ste1GluFfnx1|2xDenseLayerHid1024LayOutDr.5_Lr1e-4_500e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(156, 157):
    configs[i]['training']['epochs'] = 500
    configs[i]['sentence_linker']['input_drop'] = 0.2
    configs[i]['sentence_linker']['layers'] = [
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 1024, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': 'batch'
            }
        },
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 1024, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': 'batch'
            }
        }
    ]
    configs[i]['name'] = 'bL128_inDr.2|2xDenseLayerHid1024LayOutDr.5NormBatch_Lr1e-4_500e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(157, 158):
    configs[i]['training']['epochs'] = 500
    configs[i]['sentence_linker']['input_drop'] = 0.2
    configs[i]['sentence_linker']['layers'] = [
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 1024, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': 'layer'
            }
        },
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 1024, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': 'layer'
            }
        }
    ]
    configs[i]['name'] = 'bL128_inDr.2|2xDenseLayerHid1024LayOutDr.5NormLayer_Lr1e-4_500e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(158, 159):
    configs[i]['training']['epochs'] = 500
    configs[i]['sentence_linker']['input_drop'] = 0.2
    configs[i]['sentence_linker']['layers'] = [
        {
            'SteGluFfn': {
                'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'ste_drop': 0.5, 'ste_layers_cnt': 1
            }
        },
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None
            }
        },
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None
            }
        }
    ]
    configs[i]['name'] = 'bL128_inDr.2|Ste1GluFfnx1|2xDenseLayerDr.5_Lr1e-4_500e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(159, 160):
    configs[i]['training']['epochs'] = 500
    configs[i]['sentence_linker']['input_drop'] = 0.2
    configs[i]['sentence_linker']['layers'] = [
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None, 'highway_network': None
            }
        },
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None, 'highway_network': None
            }
        },
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None, 'highway_network': None
            }
        }
    ]
    configs[i]['name'] = 'bL128_inDr.2|3xDenseLayerDr.5_Lr1e-4_500e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(160, 161):
    configs[i]['training']['epochs'] = 500
    configs[i]['sentence_linker']['input_drop'] = 0.2
    configs[i]['sentence_linker']['layers'] = [
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 5120, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None, 'highway_network': None
            }
        },
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 5120, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None, 'highway_network': None
            }
        }
    ]
    configs[i]['name'] = 'bL128_inDr.2|2xDenseLayerHid5120Dr.5_Lr1e-4_500e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(161, 162):
    configs[i]['training']['epochs'] = 500
    configs[i]['sentence_linker']['input_drop'] = 0.2
    configs[i]['sentence_linker']['layers'] = [
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 6144, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None, 'highway_network': None
            }
        },
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 6144, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None, 'highway_network': None
            }
        }
    ]
    configs[i]['name'] = 'bL128_inDr.2|2xDenseLayerHid6144Dr.5_Lr1e-4_500e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(162, 163):
    configs[i]['training']['epochs'] = 500
    configs[i]['sentence_linker']['input_drop'] = 0.2
    configs[i]['sentence_linker']['layers'] = [
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': 'batch', 'highway_network': None
            }
        },
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': 'batch', 'highway_network': None
            }
        }
    ]
    configs[i]['name'] = 'bL128_inDr.2|2xDenseLayerDr.5NormB_Lr1e-4_500e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(163, 164):
    configs[i]['training']['epochs'] = 500
    configs[i]['sentence_linker']['input_drop'] = 0.2
    configs[i]['sentence_linker']['layers'] = [
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': 'layer', 'highway_network': None
            }
        },
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': 'layer', 'highway_network': None
            }
        }
    ]
    configs[i]['name'] = 'bL128_inDr.2|2xDenseLayerDr.5NormL_Lr1e-4_500e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(164, 165):
    configs[i]['training']['epochs'] = 500
    configs[i]['sentence_linker']['input_drop'] = 0.2
    configs[i]['sentence_linker']['layers'] = [
        {
            'DenseLayer4l': {
                'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'highway_network': None
            }
        },
        {
            'DenseLayer4l': {
                'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'highway_network': None
            }
        }
    ]
    configs[i]['name'] = 'bL128_inDr.2|2xDenseLayer4lDr.5_Lr1e-4_500e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(165, 166):
    configs[i]['training']['epochs'] = 500
    configs[i]['sentence_linker']['input_drop'] = 0.2
    configs[i]['sentence_linker']['layers'] = [
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None, 'highway_network': 'Wg(x)', 'cat_inout': False
            }
        },
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None, 'highway_network': 'Wg(x)', 'cat_inout': False
            }
        }
    ]
    configs[i]['name'] = 'bL128_inDr.2|2xDenseLayerDr.5HnetWg(x)XDr.5_Lr1e-4_500e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(166, 167):
    configs[i]['training']['epochs'] = 500
    configs[i]['sentence_linker']['input_drop'] = 0.2
    configs[i]['sentence_linker']['layers'] = [
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None, 'highway_network': None, 'cat_inout': True,
                'hidden_act': 'relu'
            }
        },
        {
            'DenseLayer': {
                'input_dim': 2*4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None, 'highway_network': None, 'cat_inout': False,
                'hidden_act': 'relu'
            }
        }
    ]
    configs[i]['name'] = 'bL128_inDr.2|2xDenseLayerDr.5CatInOut_Lr1e-4_500e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(167, 168):
    configs[i]['training']['epochs'] = 500
    configs[i]['sentence_linker']['input_drop'] = 0.2
    configs[i]['sentence_linker']['layers'] = [
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None, 'highway_network': None, 'cat_inout': True,
                'hidden_act': 'relu'
            }
        },
        {
            'DenseLayer': {
                'input_dim': 2*4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None, 'highway_network': None, 'cat_inout': False,
                'hidden_act': 'relu'
            }
        }
    ]
    configs[i]['name'] = 'bL128_inDr.2|2xDenseLayerDr.5CatYX1_Lr1e-4_500e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(168, 169):
    configs[i]['training']['epochs'] = 500
    configs[i]['sentence_linker']['input_drop'] = 0.2
    configs[i]['sentence_linker']['layers'] = [
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None, 'highway_network': None, 'cat_inout': False,
                'hidden_act': 'relu'
            }
        },
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None, 'highway_network': None, 'cat_inout': False,
                'hidden_act': 'relu'
            }
        },
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None, 'highway_network': None, 'cat_inout': False,
                'hidden_act': 'relu'
            }
        }
    ]
    configs[i]['name'] = 'bL128_inDr.2|3xDenseLayerDr.5_Lr1e-4_500e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(169, 170):
    configs[i]['training']['epochs'] = 500
    configs[i]['sentence_linker']['input_drop'] = 0.2
    configs[i]['sentence_linker']['layers'] = [
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None, 'highway_network': None, 'cat_inout': False,
                'hidden_act': 'gelu'
            }
        },
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None, 'highway_network': None, 'cat_inout': False,
                'hidden_act': 'gelu'
            }
        }
    ]
    configs[i]['name'] = 'bL128_inDr.2|2xDenseLayerDr.5Gelu_Lr1e-4_500e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(170, 171):
    configs[i]['training']['epochs'] = 500
    configs[i]['sentence_linker']['input_drop'] = 0.2
    configs[i]['sentence_linker']['layers'] = [
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None, 'highway_network': None, 'cat_inout': False,
                'hidden_act': 'mish'
            }
        },
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None, 'highway_network': None, 'cat_inout': False,
                'hidden_act': 'mish'
            }
        }
    ]
    configs[i]['name'] = 'bL128_inDr.2|2xDenseLayerDr.5Mish_Lr1e-4_500e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(171, 172):
    configs[i]['training']['epochs'] = 500
    configs[i]['sentence_linker']['input_drop'] = 0.2
    configs[i]['sentence_linker']['layers'] = [
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None, 'highway_network': None, 'cat_inout': False,
                'hidden_act': 'swish'
            }
        },
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None, 'highway_network': None, 'cat_inout': False,
                'hidden_act': 'swish'
            }
        }
    ]
    configs[i]['name'] = 'bL128_inDr.2|2xDenseLayerDr.5Swish_Lr1e-4_500e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(172, 173):
    configs[i]['training']['epochs'] = 500
    configs[i]['sentence_linker']['input_drop'] = 0.2
    configs[i]['sentence_linker']['layers'] = [
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None, 'highway_network': None, 'cat_inout': False,
                'hidden_act': 'sharkfin'
            }
        },
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None, 'highway_network': None, 'cat_inout': False,
                'hidden_act': 'sharkfin'
            }
        }
    ]
    configs[i]['name'] = 'bL128_inDr.2|2xDenseLayerDr.5Sharkfin_Lr1e-4_500e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(173, 174):
    configs[i]['training']['epochs'] = 500
    configs[i]['sentence_linker']['input_drop'] = 0.2
    configs[i]['sentence_linker']['layers'] = [
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None, 'highway_network': None, 'cat_inout': False,
                'hidden_act': 'sinrelu'
            }
        },
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None, 'highway_network': None, 'cat_inout': False,
                'hidden_act': 'sinrelu'
            }
        }
    ]
    configs[i]['name'] = 'bL128_inDr.2|2xDenseLayerDr.5Sinrelu_Lr1e-4_500e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(174, 175):
    configs[i]['training']['epochs'] = 500
    configs[i]['sentence_linker']['input_drop'] = 0.2
    configs[i]['sentence_linker']['layers'] = [
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None, 'highway_network': None, 'cat_inout': False,
                'hidden_act': 'nlrelu'
            }
        },
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None, 'highway_network': None, 'cat_inout': False,
                'hidden_act': 'nlrelu'
            }
        }
    ]
    configs[i]['name'] = 'bL128_inDr.2|2xDenseLayerDr.5Nlrelu_Lr1e-4_500e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(175, 176):
    configs[i]['training']['epochs'] = 500
    configs[i]['sentence_linker']['input_drop'] = 0.2
    configs[i]['sentence_linker']['layers'] = [
        {
            'FcLayer': {
                'input_dim': 4096, 'output_dim': 4096, 'drop': 0.2, 'act': 'relu'
            }
        },
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None, 'highway_network': None, 'cat_inout': False,
                'hidden_act': 'relu'
            }
        },
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None, 'highway_network': None, 'cat_inout': False,
                'hidden_act': 'relu'
            }
        }
    ]
    configs[i]['name'] = 'bL128_inDr.2|FcReluDr.2|2xDenseLayerDr.5_Lr1e-4_500e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(176, 177):
    configs[i]['training']['epochs'] = 500
    configs[i]['training']['l1_loss'] = 1e-4
    configs[i]['sentence_linker']['input_drop'] = 0.2
    configs[i]['sentence_linker']['layers'] = [
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None, 'highway_network': None, 'cat_inout': False,
                'hidden_act': 'relu'
            }
        },
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None, 'highway_network': None, 'cat_inout': False,
                'hidden_act': 'relu'
            }
        }
    ]
    configs[i]['name'] = 'bL128_inDr.2|2xDenseLayerDr.5_l1Loss1e-4_Lr1e-4_500e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(177, 178):
    configs[i]['training']['epochs'] = 500
    configs[i]['training']['l1_loss'] = 1e-8
    configs[i]['sentence_linker']['input_drop'] = 0.2
    configs[i]['sentence_linker']['layers'] = [
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None, 'highway_network': None, 'cat_inout': False,
                'hidden_act': 'relu'
            }
        },
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None, 'highway_network': None, 'cat_inout': False,
                'hidden_act': 'relu'
            }
        }
    ]
    configs[i]['name'] = 'bL128_inDr.2|2xDenseLayerDr.5_l1Loss1e-8_Lr1e-4_500e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(178, 179):
    configs[i]['training']['epochs'] = 500
    configs[i]['sentence_linker']['input_drop'] = 0.2
    configs[i]['sentence_linker']['layers'] = [
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.7, 'drop_pos': 'output', 'norm': None, 'highway_network': None, 'cat_inout': False,
                'hidden_act': 'relu'
            }
        },
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.7, 'drop_pos': 'output', 'norm': None, 'highway_network': None, 'cat_inout': False,
                'hidden_act': 'relu'
            }
        }
    ]
    configs[i]['name'] = 'bL128_inDr.2|2xDenseLayerDr.7_Lr1e-4_500e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(179, 180):
    configs[i]['training']['epochs'] = 500
    configs[i]['sentence_linker']['input_drop'] = 0.2
    configs[i]['sentence_linker']['layers'] = [
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None, 'highway_network': 'Wg(SteGluFfn(x))', 'cat_inout': False,
                'hidden_act': 'relu'
            }
        },
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None, 'highway_network': 'Wg(SteGluFfn(x))', 'cat_inout': False,
                'hidden_act': 'relu'
            }
        }
    ]
    configs[i]['name'] = 'bL128_inDr.2|2xDenseLayerDr.5SteGluFfnGate_Lr1e-4_500e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(180, 181):
    configs[i]['training']['epochs'] = 500
    configs[i]['sentence_linker']['input_drop'] = 0.2
    configs[i]['sentence_linker']['layers'] = [
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None, 'highway_network': 'Wg(x,y)*y', 'cat_inout': False,
                'hidden_act': 'relu'
            }
        },
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None, 'highway_network': 'Wg(x,y)*y', 'cat_inout': False,
                'hidden_act': 'relu'
            }
        }
    ]
    configs[i]['name'] = 'bL128_inDr.2|2xDenseLayerDr.5Wg(x,y)*y_Lr1e-4_500e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(181, 182):
    configs[i]['training']['epochs'] = 500
    configs[i]['sentence_linker']['input_drop'] = 0.2
    configs[i]['sentence_linker']['layers'] = [
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None, 'highway_network': 'Wg(x,y)*y', 'cat_inout': False,
                'hidden_act': 'relu'
            }
        },
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None, 'highway_network': 'Wg(x,y)*y', 'cat_inout': False,
                'hidden_act': 'relu'
            }
        }
    ]
    configs[i]['name'] = 'bL128_inDr.2|2xDenseLayerDr.5Wg(x,y)*yDr.2.5_Lr1e-4_500e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(182, 183):
    configs[i]['training']['epochs'] = 500
    configs[i]['sentence_linker']['input_drop'] = 0.2
    configs[i]['sentence_linker']['layers'] = [
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None, 'highway_network': 'Wg(x,y)*y', 'cat_inout': False,
                'hidden_act': 'relu'
            }
        },
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None, 'highway_network': None, 'cat_inout': False,
                'hidden_act': 'relu'
            }
        }
    ]
    configs[i]['name'] = 'bL128_inDr.2|2xDenseLayerDr.5.1xWg(x,y)*yDr.2.5_Lr1e-4_500e_'+str(i)
# -----------------------------------------------------------------------------
for i in range(183, 184):
    configs[i]['training']['epochs'] = 500
    configs[i]['sentence_linker']['input_drop'] = 0.2
    configs[i]['sentence_linker']['layers'] = [
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None, 'highway_network': 'Ste(x,y)*y', 'cat_inout': False,
                'hidden_act': 'relu'
            }
        },
        {
            'DenseLayer': {
                'input_dim': 4096, 'hidden_dim': 4096, 'output_dim': 4096, 'hid_drop': 0.5, 'drop_pos': 'output', 'norm': None, 'highway_network': 'Ste(x,y)*y', 'cat_inout': False,
                'hidden_act': 'relu'
            }
        }
    ]
    configs[i]['name'] = 'bL128_inDr.2|2xDenseLayerDr.5Ste(x,y)*yDr.5_Lr1e-4_500e_'+str(i)
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