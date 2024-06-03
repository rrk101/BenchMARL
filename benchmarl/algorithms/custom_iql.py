#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass, MISSING
from typing import Dict, Iterable, Tuple, Type, List

from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec
from torchrl.modules import EGreedyModule, QValueModule
from torchrl.objectives import DQNLoss, LossModule, ValueEstimators

from torchrl.envs import Compose, Transform
from torchrl.data.replay_buffers import RandomSampler, SamplerWithoutReplacement, PrioritizedSampler
from torchrl.data import (
    DiscreteTensorSpec,
    LazyTensorStorage,
    OneHotDiscreteTensorSpec,
    ReplayBuffer,
    TensorDictReplayBuffer,
    TensorDictPrioritizedReplayBuffer,
)

from benchmarl.algorithms.common import Algorithm, AlgorithmConfig
from benchmarl.models.common import ModelConfig


class Custom_Iql(Algorithm):
    """Independent Q Learning (from `https://www.semanticscholar.org/paper/Multi-Agent-Reinforcement-Learning%3A-Independent-Tan/59de874c1e547399b695337bcff23070664fa66e <https://www.semanticscholar.org/paper/Multi-Agent-Reinforcement-Learning%3A-Independent-Tan/59de874c1e547399b695337bcff23070664fa66e>`__).

    Args:
        loss_function (str): loss function for the value discrepancy. Can be one of "l1", "l2" or "smooth_l1".
        delay_value (bool): whether to separate the target value networks from the value networks used for
            data collection.
        
        PER_alpha(float): the alpha for prioritized experience replay buffer
        PER_beta(float): the beta for prioritized experience replay buffer


    """

    def __init__(
            self,
            delay_value: bool, 
            loss_function: str,
            PER_alpha: float,
            PER_beta: float,
            **kwargs
        ):
        super().__init__(**kwargs)

        self.delay_value = delay_value
        self.loss_function = loss_function

        self.PER_alpha = PER_alpha
        self.PER_beta = PER_beta
    
    # my additions
       # Overriding the get_replay_buffer from parent class Algorithm
    def get_replay_buffer(
        self, group: str, transforms: List[Transform] = None
    ) -> ReplayBuffer:
        """
        Get the ReplayBuffer for a specific group.
        This function will check ``self.on_policy`` and create the buffer accordingly

        Args:
            group (str): agent group of the loss and updater
            transforms (optional, list of Transform): Transforms to apply to the replay buffer ``.sample()`` call

        Returns: ReplayBuffer the group
        """
        # print("I am in get_replay_buffer of customippo")

        memory_size = self.experiment_config.replay_buffer_memory_size(self.on_policy)
        sampling_size = self.experiment_config.train_minibatch_size(self.on_policy)
        storing_device = self.device
        # sampler = SamplerWithoutReplacement() if self.on_policy else RandomSampler()
        # adding alpha and beta by myself, better if sourced from somewhere and put on algorithm config
        # sampler = PrioritizedSampler(max_capacity=sampling_size, alpha=0.6, beta=0.4) if self.on_policy else RandomSampler()
        # return TensorDictReplayBuffer(
        #     storage=LazyTensorStorage(memory_size, device=storing_device),
        #     sampler=sampler,
        #     batch_size=sampling_size,
        #     priority_key=(group, "td_error"),
        #     transform=Compose(*transforms) if transforms is not None else None,
        # )
        return TensorDictPrioritizedReplayBuffer(
            storage=LazyTensorStorage(memory_size, device=storing_device),
            batch_size=sampling_size,
            alpha=self.PER_alpha,
            beta=self.PER_beta,
            transform=Compose(*transforms) if transforms is not None else None,
        )
    

    #############################
    # Overridden abstract methods
    #############################

    def _get_loss(
        self, group: str, policy_for_loss: TensorDictModule, continuous: bool
    ) -> Tuple[LossModule, bool]:
        if continuous:
            raise NotImplementedError("Iql is not compatible with continuous actions.")
        else:
            # Loss
            loss_module = DQNLoss(
                policy_for_loss,
                delay_value=self.delay_value,
                loss_function=self.loss_function,
                action_space=self.action_spec[group, "action"],
            )
            loss_module.set_keys(
                reward=(group, "reward"),
                action=(group, "action"),
                done=(group, "done"),
                terminated=(group, "terminated"),
                action_value=(group, "action_value"),
                value=(group, "chosen_action_value"),
                priority=(group, "td_error"),
            )
            loss_module.make_value_estimator(
                ValueEstimators.TD0, gamma=self.experiment_config.gamma
            )

            return loss_module, True

    def _get_parameters(self, group: str, loss: LossModule) -> Dict[str, Iterable]:
        return {"loss": loss.parameters()}

    def _get_policy_for_loss(
        self, group: str, model_config: ModelConfig, continuous: bool
    ) -> TensorDictModule:
        n_agents = len(self.group_map[group])
        logits_shape = [
            *self.action_spec[group, "action"].shape,
            self.action_spec[group, "action"].space.n,
        ]

        actor_input_spec = CompositeSpec(
            {group: self.observation_spec[group].clone().to(self.device)}
        )

        actor_output_spec = CompositeSpec(
            {
                group: CompositeSpec(
                    {"action_value": UnboundedContinuousTensorSpec(shape=logits_shape)},
                    shape=(n_agents,),
                )
            }
        )
        actor_module = model_config.get_model(
            input_spec=actor_input_spec,
            output_spec=actor_output_spec,
            agent_group=group,
            input_has_agent_dim=True,
            n_agents=n_agents,
            centralised=False,
            share_params=self.experiment_config.share_policy_params,
            device=self.device,
            action_spec=self.action_spec,
        )
        if self.action_mask_spec is not None:
            action_mask_key = (group, "action_mask")
        else:
            action_mask_key = None

        value_module = QValueModule(
            action_value_key=(group, "action_value"),
            action_mask_key=action_mask_key,
            out_keys=[
                (group, "action"),
                (group, "action_value"),
                (group, "chosen_action_value"),
            ],
            spec=self.action_spec[group, "action"],
            action_space=None,
        )

        return TensorDictSequential(actor_module, value_module)

    def _get_policy_for_collection(
        self, policy_for_loss: TensorDictModule, group: str, continuous: bool
    ) -> TensorDictModule:
        if self.action_mask_spec is not None:
            action_mask_key = (group, "action_mask")
        else:
            action_mask_key = None

        greedy = EGreedyModule(
            annealing_num_steps=self.experiment_config.get_exploration_anneal_frames(
                self.on_policy
            ),
            action_key=(group, "action"),
            spec=self.action_spec[(group, "action")],
            action_mask_key=action_mask_key,
            eps_init=self.experiment_config.exploration_eps_init,
            eps_end=self.experiment_config.exploration_eps_end,
        )
        return TensorDictSequential(*policy_for_loss, greedy)

    def process_batch(self, group: str, batch: TensorDictBase) -> TensorDictBase:
        keys = list(batch.keys(True, True))
        group_shape = batch.get(group).shape

        nested_done_key = ("next", group, "done")
        nested_terminated_key = ("next", group, "terminated")
        nested_reward_key = ("next", group, "reward")

        if nested_done_key not in keys:
            batch.set(
                nested_done_key,
                batch.get(("next", "done")).unsqueeze(-1).expand((*group_shape, 1)),
            )
        if nested_terminated_key not in keys:
            batch.set(
                nested_terminated_key,
                batch.get(("next", "terminated"))
                .unsqueeze(-1)
                .expand((*group_shape, 1)),
            )

        if nested_reward_key not in keys:
            batch.set(
                nested_reward_key,
                batch.get(("next", "reward")).unsqueeze(-1).expand((*group_shape, 1)),
            )

        return batch

    #####################
    # Custom new methods
    #####################


@dataclass
class Custom_IqlConfig(AlgorithmConfig):
    """Configuration dataclass for :class:`~benchmarl.algorithms.Iql`."""

    delay_value: bool = MISSING
    loss_function: str = MISSING
    #my additions
    PER_alpha: float = MISSING
    PER_beta: float = MISSING

    @staticmethod
    def associated_class() -> Type[Algorithm]:
        return Custom_Iql

    @staticmethod
    def supports_continuous_actions() -> bool:
        return False

    @staticmethod
    def supports_discrete_actions() -> bool:
        return True

    @staticmethod
    def on_policy() -> bool:
        return False
