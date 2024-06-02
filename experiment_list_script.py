from benchmarl.algorithms import IppoConfig, Custom_IppoConfig
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig

# Loads from "benchmarl/conf/experiment/base_experiment.yaml"
experiment_config = ExperimentConfig.get_from_yaml()
# Loads from "benchmarl/conf/task/vmas/balance.yaml"
task = VmasTask.DISPERSION.get_from_yaml()
# Loads from "benchmarl/conf/algorithm/mappo.yaml"
algorithm_config = IppoConfig.get_from_yaml()
# Loads from "benchmarl/conf/model/layers/mlp.yaml"
model_config = MlpConfig.get_from_yaml()
critic_model_config = MlpConfig.get_from_yaml()

experiment_config.max_n_iters = 2
experiment_config.loggers = []
global_seed = 0

experiment = Experiment(
    task=task,
    algorithm_config=algorithm_config,
    model_config=model_config,
    critic_model_config=critic_model_config,
    seed=global_seed,
    config=experiment_config,
)
experiment.run()

# 2nd Experiment 

algorithm_config = Custom_IppoConfig.get_from_yaml()
experiment_config.share_experiences = True
share_experiences_freqs = [1, 5, 10, 50, 100]
communication_bandwidths = [0.001, 0.01, 0.1, 0.5, 1.0]
PER_params = [(0.1, 0.0), (0.5, 0.4), (0.6, 0.4), (0.7, 0.5), (1.0, 0.7)]
PER_params = PER_params[::-1] #reverse the list

for freq in share_experiences_freqs:
    for bandwidth in communication_bandwidths:
        for alpha, beta in PER_params:
            experiment_config.share_experiences_freq = freq
            experiment_config.share_experiences_bandwidth = bandwidth
            algorithm_config.PER_alpha = alpha
            algorithm_config.PER_beta = beta

            experiment = Experiment(
                task=task,
                algorithm_config=algorithm_config,
                model_config=model_config,
                critic_model_config=critic_model_config,
                seed=global_seed,
                config=experiment_config,
            )
            experiment.run()
