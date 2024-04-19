"""This file configures the zoo using python scripts instead of yaml files."""

from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from mqt.predictor.rl.torch_layers import CustomCombinedExtractor

hyperparams = {
    "PredictorEnv-v0": dict(
        # n_envs: The number of environments to run in parallel. 
        # This can speed up training by allowing the agent to gather experience from multiple environments at once.
        n_envs = 1,

        # n_timesteps: The total number of steps to be taken by the agent in the environment during the training process.
        # A step typically corresponds to the agent taking an action in the environment and receiving the resulting observation and reward.
        n_timesteps = 10000,

        # n_steps: The number of steps to run for each environment per update (i.e., batch size).
        # For example, if n_steps is 5, then the agent will take 5 steps in the environment, then update, then take another 5 steps, and so on.
        # NOTE: this is not the max. num of steps! it doesnt matter when the episode ends
        n_steps = 64,

        # batch_size: The number of experiences to use in each update. Number of samples taken out of the replay buffer.
        # Larger batch sizes can lead to more stable updates, but they also require more memory and can slow down training.
        batch_size = 16,

        # n_epochs: The number of times to iterate through the entire batch during the training process.
        # For example, if n_epochs is 2, then the agent will go through the entire batch of experiences twice during each update.
        # This is a form of "replay" that can help the agent learn from its experiences more effectively.
        n_epochs = 2,

        gamma=0.98,
        learning_rate=1e-03,

        callback = "rl_zoo3.callbacks.CustomCallback",

        env_kwargs = dict(
            device_name = "ionq_harmony", #ibm_washington", #"ionq_aria1", # "ionq_harmony",
            reward_function = "expected_fidelity",
            features = [
                "num_qubits",
                "depth",
                "program_communication",
                "critical_depth",
                "entanglement_ratio",
                "parallelism",
                "liveness",

                #"directed_program_communication",
                #"singleQ_gates_per_layer",
                #"multiQ_gates_per_layer",

                #"circuit",
                #"graph_edge_index",
                #"graph_x",
                #"graph_edge_attr",
            ],
        ),

        policy=MaskableMultiInputActorCriticPolicy,
        policy_kwargs=dict(
            features_extractor_class=CustomCombinedExtractor,
            features_extractor_kwargs=dict(
                cnn_output_dim = 64,
                #normalized_image = False,
                #hidden_dim = 256,
                #out_channels = 64, # 1 == no CNN
                #kernel_size = 3,
                #stride = 1,
                #padding = 1,
                #num_layers = 1,
            ),
       ),
    ),
}
