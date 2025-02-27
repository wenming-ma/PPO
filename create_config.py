import json

config = {
    "env": "CartPole-v0",
    "input_net_type": "MLP",
    "grayscale_transform": False,
    "markov_length": 4,
    "dilation": 1,
    "total_num_state_transitions": 200000,
    "param_sharing": True,

    "epochs": 5,
    "parallel_agents": 8,
    "trajectory_length": 600,
    "discount_factor": 0.99,
    "batch_size": 64,

    "learning_rate_pol": 0.0003,
    "learning_rate_val": 0.0003,

    "clipping_parameter": {
        "decay_type": "linear",
        "initial_value": 0.2,
        "min_value": 0.01,
        "verbose": True
    },

    "entropy_contrib_factor": 0.01,
    "vf_contrib_factor": 1.0,
    
    "show_final_demo": True,
    "frame_duration": 0.08,
    "max_render_time_steps": 500,

    "deterministic_eval": True,
    "stochastic_eval": True,
    "intermediate_eval_steps": 200,
    "time_steps_extensive_eval": 10000,

    "network_structure": [
        64,
        128,
        64
    ],
    "nonlinearity": "relu"
}

with open('pole_config.py', 'w', encoding='utf-8') as f:
    f.write(str(config)) 