# This file contains experiment configurations that will be run by the script walk_visualisation.py

# Example model ensemble configurations. Every model in the ensemble can have it's own
# walk rate, energy valley width and energy valley depth.
#            model, select_idx, (lr, sigma, depth)
config_b = [('male', False, (5.9, 12.0, 22.0)),
            ('male', True, (5.9, 16.0, 16.0)),
            ('old', True, (1.2, 1.6, 1.6)),
            ('wavy_hair', True, (4.8, 2.8, 2.8)),
            ('wavy_hair', False, (4.8, 2.7, 2.7)), ]
config_b_no_walk = [('male', False, (0.0, 12.0, 22.0)),
                    ('male', True, (0.0, 16.0, 16.0)),
                    ('old', True, (0.0, 1.6, 1.6)),
                    ('wavy_hair', True, (0.0, 2.8, 2.8)),
                    ('wavy_hair', False, (0.0, 2.7, 2.7)), ]
config_b_small_test_set = [('male', False, (5.0, 10.0, 22.0)),
                           ('male', True, (6.0, 11.0, 18.0)),
                           ('old', True, (1.0, 1.0, 1.1)),
                           ('wavy_hair', True, (4.5, 2.4, 2.4)),
                           ('wavy_hair', False, (5.0, 2.4, 2.2)), ]
config_c = [('male', False, (0.7, 1.1, 0.7)),
            ('male', True, (0.7, 1.2, 0.8)),
            ('wavy_hair', True, (0.7, 1.5, 1.0)),
            ('wavy_hair', False, (0.5, 1.6, 1.6)), ]
config_a = [('male', False, (0.1, 0.25, 0.7)),
            ('male', True, (0.1, 0.25, 0.7)),
            ('wavy_hair', True, (0.05, 1.0, 2.5)),
            ('wavy_hair', False, (0.05, 1.0, 2.5)), ]

config_vars = [[(model, cond, (lr * k, sigma * i, depth * j)) for model, cond, (lr, sigma, depth) in config_c]
               for i in [0.3, 0.8, 1.0, 2]
               for j in [0.3, 0.8, 1.0, 2]
               for k in [0.5, 1.0, 2]]
config_male = [('male', False, (1.0, 13.0, 13.0)), ]
config_male_strong = [('male', False, (1.0, 22.0, 22.0)), ]

search = None
# example experiment parameters - add multiple tuples for multiple experiments
# model selection config, scales, walk rate, valley depth, valley sigma, noise, num steps, test_size
search = [
    (config_male, (2,), 5.0, 2.0, 1.5, 0.005, 50, 8),
    (config_male, (2,), 5.0, 2.0, 1.5, 0.03, 50, 8),
    (config_male_strong, (2,), 5.0, 2.0, 1.5, 0.1, 50, 8),
]
'''search = [
    (conf, (2,), 5.0, 2.0, 1.5, n, 50, 8)
    for conf in [config_a]
    for n in (0.01,)
]'''
'''search = [
    (config_b_no_walk, (2,), 5.0, 2.0, 1.5, 0.001, 1, 200),
]'''

# The embedded space variations a, b, c described in the thesis, are defined here as follows:
# a:(model_version=0, scales=(2,)),
# b:(model_version=1, scales=(2,)),
# c:(model_version=0, scales=(0,1,2))
# where model_version is a commandline argument for the script walk_visualisation.py
