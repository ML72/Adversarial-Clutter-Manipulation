from typing import List, Optional, Union

import atexit
import sys
import tqdm

import torch
import torch.nn as nn

import numpy as np
import os

from skrl import config, logger
from skrl.agents.torch import Agent
from skrl.envs.wrappers.torch import Wrapper
from skrl.agents.torch.ppo import PPO
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.memories.torch import RandomMemory


# TODO: find better way to import this without hard coding it here
# although it seems like this is hard coded at the top of the files for SKRL demos too
# define the shared model
class SharedModel(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction, role="policy")
        DeterministicMixin.__init__(self, clip_actions, role="value")

        # shared layers/network
        self.net = nn.Sequential(nn.Linear(self.num_observations, 32),
                                 nn.ELU(),
                                 nn.Linear(32, 32),
                                 nn.ELU())

        # separated layers ("policy")
        self.mean_layer = nn.Linear(32, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

        # separated layer ("value")
        self.value_layer = nn.Linear(32, 1)

        # Shared value is 0
        self._shared_output = None

    # override the .act(...) method to disambiguate its call
    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    # forward the input to compute model output according to the specified role
    def compute(self, inputs, role):
        if role == "policy":
            # save shared layers/network output to perform a single forward-pass
            self._shared_output = self.net(inputs["states"])
            return self.mean_layer(self._shared_output), self.log_std_parameter, {}
        elif role == "value":
            # use saved shared layers/network output to perform a single forward-pass, if it was saved
            shared_output = self.net(inputs["states"]) if self._shared_output is None else self._shared_output
            self._shared_output = None  # reset saved shared output to prevent the use of erroneous data in subsequent steps
            return self.value_layer(shared_output), {}

def generate_equally_spaced_scopes(num_envs: int, num_simultaneous_agents: int) -> List[int]:
    """Generate a list of equally spaced scopes for the agents

    :param num_envs: Number of environments
    :type num_envs: int
    :param num_simultaneous_agents: Number of simultaneous agents
    :type num_simultaneous_agents: int

    :raises ValueError: If the number of simultaneous agents is greater than the number of environments

    :return: List of equally spaced scopes
    :rtype: List[int]
    """
    scopes = [int(num_envs / num_simultaneous_agents)] * num_simultaneous_agents
    if sum(scopes):
        scopes[-1] += num_envs - sum(scopes)
    else:
        raise ValueError(
            f"The number of simultaneous agents ({num_simultaneous_agents}) is greater than the number of environments ({num_envs})"
        )
    return scopes


class Trainer:
    def __init__(
        self,
        env: Wrapper,
        agents: Union[Agent, List[Agent]],
        agents_scope: Optional[List[int]] = None,
        cfg: Optional[dict] = None,
    ) -> None:
        """Base class for trainers

        :param env: Environment to train on
        :type env: skrl.envs.wrappers.torch.Wrapper
        :param agents: Agents to train
        :type agents: Union[Agent, List[Agent]]
        :param agents_scope: Number of environments for each agent to train on (default: ``None``)
        :type agents_scope: tuple or list of int, optional
        :param cfg: Configuration dictionary (default: ``None``)
        :type cfg: dict, optional
        """
        self.cfg = cfg if cfg is not None else {}
        self.env = env
        self.agents = agents
        self.agents_scope = agents_scope if agents_scope is not None else []

        # get configuration
        self.timesteps = self.cfg.get("timesteps", 0)
        self.headless = self.cfg.get("headless", False)
        self.disable_progressbar = self.cfg.get("disable_progressbar", False)
        self.close_environment_at_exit = self.cfg.get("close_environment_at_exit", True)
        self.environment_info = self.cfg.get("environment_info", "episode")
        self.stochastic_evaluation = self.cfg.get("stochastic_evaluation", False)

        self.initial_timestep = 0

        # setup agents
        self.num_simultaneous_agents = 0
        self._setup_agents()

        # set local variables
        self.positioning_strategy = self._isaaclab_env().cfg.positioning_strategy
        self.adversary_active = self.positioning_strategy == "pure_adversary" or self.positioning_strategy == "regret_adversary"
        self.log_adversary = "adversary" in self.positioning_strategy
        self.regret_rollouts = 5

        # setup adversary
        self.adversary_num_inputs = 4 # arbitrary number of inputs, is a noise vector to condition on
        num_outputs = (self._isaaclab_env().num_clutter_objects + 1) * 3 # clutter + main object
        adversary_shared_model = SharedModel(self.adversary_num_inputs, num_outputs, device=env.device)
        models = { # TODO: not sure if this is supposed to be shared
            "policy": adversary_shared_model,
            "value": adversary_shared_model
        }

        adversary_rollouts = 1 if self.positioning_strategy == "regret_adversary" else self.regret_rollouts
        adversary_memsize = 25 // self.regret_rollouts if self.positioning_strategy == "regret_adversary" else 25
        adversary_cfg = {
            "rollouts": adversary_rollouts, # make it fair
            "learning_starts": adversary_memsize, # explore a bit more before learning
            "memory_size": adversary_memsize, # passed into RandomMemory manually, must be <= learning_starts
            "learning_rate": 1e-4
        }

        self.adversary = PPO(
            models=models,
            device=env.device,
            observation_space=self.adversary_num_inputs,
            action_space=num_outputs,
            memory=RandomMemory(
                num_envs=self.env.num_envs,
                memory_size=adversary_cfg["memory_size"],
                device=env.device
            ),
            cfg=adversary_cfg
        )
        self.adversary.init()

        # register environment closing if configured
        if self.close_environment_at_exit:

            @atexit.register
            def close_env():
                logger.info("Closing environment")
                self.env.close()
                logger.info("Environment closed")

        # update trainer configuration to avoid duplicated info/data in distributed runs
        if config.torch.is_distributed:
            if config.torch.rank:
                self.disable_progressbar = True

    def __str__(self) -> str:
        """Generate a string representation of the trainer

        :return: Representation of the trainer as string
        :rtype: str
        """
        string = f"Trainer: {self}"
        string += f"\n  |-- Number of parallelizable environments: {self.env.num_envs}"
        string += f"\n  |-- Number of simultaneous agents: {self.num_simultaneous_agents}"
        string += "\n  |-- Agents and scopes:"
        if self.num_simultaneous_agents > 1:
            for agent, scope in zip(self.agents, self.agents_scope):
                string += f"\n  |     |-- agent: {type(agent)}"
                string += f"\n  |     |     |-- scope: {scope[1] - scope[0]} environments ({scope[0]}:{scope[1]})"
        else:
            string += f"\n  |     |-- agent: {type(self.agents)}"
            string += f"\n  |     |     |-- scope: {self.env.num_envs} environment(s)"
        return string

    def _setup_agents(self) -> None:
        """Setup agents for training

        :raises ValueError: Invalid setup
        """
        # validate agents and their scopes
        if type(self.agents) in [tuple, list]:
            # single agent
            if len(self.agents) == 1:
                self.num_simultaneous_agents = 1
                self.agents = self.agents[0]
                self.agents_scope = [1]
            # parallel agents
            elif len(self.agents) > 1:
                self.num_simultaneous_agents = len(self.agents)
                # check scopes
                if not len(self.agents_scope):
                    logger.warning("The agents' scopes are empty, they will be generated as equal as possible")
                    self.agents_scope = [int(self.env.num_envs / len(self.agents))] * len(self.agents)
                    if sum(self.agents_scope):
                        self.agents_scope[-1] += self.env.num_envs - sum(self.agents_scope)
                    else:
                        raise ValueError(
                            f"The number of agents ({len(self.agents)}) is greater than the number of parallelizable environments ({self.env.num_envs})"
                        )
                elif len(self.agents_scope) != len(self.agents):
                    raise ValueError(
                        f"The number of agents ({len(self.agents)}) doesn't match the number of scopes ({len(self.agents_scope)})"
                    )
                elif sum(self.agents_scope) != self.env.num_envs:
                    raise ValueError(
                        f"The scopes ({sum(self.agents_scope)}) don't cover the number of parallelizable environments ({self.env.num_envs})"
                    )
                # generate agents' scopes
                index = 0
                for i in range(len(self.agents_scope)):
                    index += self.agents_scope[i]
                    self.agents_scope[i] = (index - self.agents_scope[i], index)
            else:
                raise ValueError("A list of agents is expected")
        else:
            self.num_simultaneous_agents = 1

    def train(self) -> None:
        """Train the agents

        :raises NotImplementedError: Not implemented
        """
        raise NotImplementedError

    def eval(self) -> None:
        """Evaluate the agents

        :raises NotImplementedError: Not implemented
        """
        raise NotImplementedError

    def single_agent_train(self) -> None:
        """Train agent

        This method executes the following steps in loop:

        - Pre-interaction
        - Compute actions
        - Interact with the environments
        - Render scene
        - Record transitions
        - Post-interaction
        - Reset environments
        """
        assert self.num_simultaneous_agents == 1, "This method is not allowed for simultaneous agents"
        assert self.env.num_agents == 1, "This method is not allowed for multi-agents"

        # useful constants
        NUM_ENVS = self.env.num_envs
        ADVERSARY_ACTION_SPACE = self._isaaclab_env().adversary_action.shape[-1]
        MAX_EPISODE_LENGTH = self._isaaclab_env().max_episode_length

        # utility function to get an adversary action given the sampling strategy
        def get_adversary_action(
            rand_state: torch.Tensor,
            device: torch.device,
            timestep=0,
            regret_trials=0,
            rewards=None,
            prev_action=None
        ) -> torch.Tensor:
            result_action = None
            if self.positioning_strategy == "domain_rand":
                # Randomly sample every action dimension from -1 to 1
                result_action = torch.rand((NUM_ENVS, ADVERSARY_ACTION_SPACE), device=device) * 2 - 1
            elif self.positioning_strategy == "domain_rand_restricted":
                # Randomly sample every action dimension from a subrange smaller than -1 to 1
                result_action = torch.rand((NUM_ENVS, ADVERSARY_ACTION_SPACE), device=device)
                result_action[:,0] = result_action[:,0] * 2 - 1 # y direction is stretched to range [-1,1]
                result_action[:,1] = result_action[:,1] # x direction is unchaged, in range [0,1]
            elif self.positioning_strategy == "boosting_adversary":
                # Boost samples that the agent performs poorly on
                result_action = torch.rand((NUM_ENVS, ADVERSARY_ACTION_SPACE), device=device) * 2 - 1
                if timestep > 0:
                    # Perturb and re-learn from past action if agent performed poorly
                    if rewards is not None and prev_action is not None:
                        mask = (rewards < rewards.median()).flatten()
                        if torch.sum(mask).item() > 0:
                            result_action[mask] = prev_action[mask] + result_action[mask] * 0.05
            elif self.positioning_strategy == "pure_adversary":
                # Pre interaction for the adversary
                self.adversary.pre_interaction(
                    timestep=((timestep+1) // MAX_EPISODE_LENGTH),
                    timesteps=(self.timesteps // MAX_EPISODE_LENGTH)
                )

                # Choose an action from a purely adversarial network
                with torch.no_grad():
                    result_action = self.adversary.act(
                        rand_state,
                        timestep=((timestep+1) // MAX_EPISODE_LENGTH),
                        timesteps=(self.timesteps // MAX_EPISODE_LENGTH)
                    )[0]
            elif self.positioning_strategy == "regret_adversary":
                if regret_trials <= 0:
                    # Pre interaction for the adversary
                    self.adversary.pre_interaction(
                        timestep=((timestep+1) // MAX_EPISODE_LENGTH // self.regret_rollouts),
                        timesteps=(self.timesteps // MAX_EPISODE_LENGTH // self.regret_rollouts)
                    )

                    # Choose an action from a purely adversarial network
                    with torch.no_grad():
                        result_action = self.adversary.act(
                            rand_state,
                            timestep=((timestep+1) // MAX_EPISODE_LENGTH // self.regret_rollouts),
                            timesteps=(self.timesteps // MAX_EPISODE_LENGTH // self.regret_rollouts)
                        )[0]
                else:
                    result_action = prev_action
            else:
                raise ValueError(f"Invalid positioning strategy: {self.positioning_strategy}")
            return result_action

        # reset env
        rand_state = torch.randn((NUM_ENVS, self.adversary_num_inputs), device=self.env.device)
        adversary_action = get_adversary_action(rand_state, self.env.device, timestep=0)
        episode_rewards = torch.zeros((NUM_ENVS, 1), device=self.env.device)

        self._isaaclab_env().adversary_action = adversary_action
        states, infos = self.env.reset()

        # set up regret adversary state, if applicable
        regret_trials = self.regret_rollouts
        max_adversary_rewards = torch.zeros((NUM_ENVS, 1), device=self.env.device)
        total_adversary_rewards = torch.zeros((NUM_ENVS, 1), device=self.env.device)
        total_adversary_penalty = torch.zeros((NUM_ENVS, 1), device=self.env.device)

        # start training loop
        adversary_action_log = []
        adversary_reward_log = []
        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar, file=sys.stdout
        ):
            # take next action from adversary if at the end of an episode
            if timestep % MAX_EPISODE_LENGTH == MAX_EPISODE_LENGTH - 1:
                rand_state = torch.randn((NUM_ENVS, self.adversary_num_inputs), device=self.env.device)
                adversary_action = get_adversary_action(
                    rand_state,
                    self.env.device,
                    timestep=timestep,
                    regret_trials=regret_trials,
                    rewards=rewards,
                    prev_action=adversary_action
                )
                regret_trials = self.regret_rollouts
                self._isaaclab_env().adversary_action = adversary_action

                # reset regret trials if applicable
                if self.positioning_strategy == "regret_adversary":
                    regret_trials -= 1
                    if regret_trials < 0:
                        regret_trials += self.regret_rollouts

                # reset episode rewards
                episode_rewards = torch.zeros((NUM_ENVS, 1), device=self.env.device)

            # pre-interaction
            self.agents.pre_interaction(timestep=timestep, timesteps=self.timesteps)

            with torch.no_grad():
                # compute actions
                actions = self.agents.act(states, timestep=timestep, timesteps=self.timesteps)[0]

                # step the environments
                next_states, rewards, terminated, truncated, infos = self.env.step(actions)
                
                # update episode rewards
                # skip reward from last episode
                if timestep % MAX_EPISODE_LENGTH != MAX_EPISODE_LENGTH - 1:
                    episode_rewards += rewards

                # render scene
                if not self.headless:
                    self.env.render()

                # record the environments' transitions
                self.agents.record_transition(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    infos=infos,
                    timestep=timestep,
                    timesteps=self.timesteps,
                )

                # log environment info
                if self.environment_info in infos:
                    for k, v in infos[self.environment_info].items():
                        if isinstance(v, torch.Tensor) and v.numel() == 1:
                            self.agents.track_data(f"Info / {k}", v.item())
            
            # agents post interaction
            self.agents.post_interaction(timestep=timestep, timesteps=self.timesteps)

            # update adversary at end of episode
            # called at second last episode step to allow for adversary action to be taken at the last step
            # due to software engineering limitations, we skip the last step of the episode, which is negligible
            states = next_states
            if timestep % MAX_EPISODE_LENGTH == MAX_EPISODE_LENGTH - 2:
                # reset_env_ids is currently not used but it should be equivalent to range(NUM_ENVS)
                reset_env_ids = self.env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
                
                if self.adversary_active:
                    # update adversary
                    with torch.no_grad():                    
                        # compute range penalty: penalizes for action values outside [-1,1] range
                        range_penalty = torch.sum(torch.maximum(
                            (adversary_action ** 2) - 1,
                            torch.zeros(adversary_action.shape, device=self.env.device)
                        ), dim=1, keepdim=True)

                        # compute reward, split by cases for different agents
                        if self.positioning_strategy == "regret_adversary":
                            adversary_rewards = (-1 * episode_rewards) # - range_penalty

                            # update regret adversary state
                            if regret_trials >= self.regret_rollouts - 1:
                                max_adversary_rewards = adversary_rewards.clone()
                            else:
                                max_adversary_rewards = torch.maximum(max_adversary_rewards, adversary_rewards)
                            total_adversary_rewards += adversary_rewards
                            total_adversary_penalty += range_penalty

                            # unsure about assignment of states and next_states
                            if regret_trials <= 0:
                                mean_adversary_rewards = total_adversary_rewards / self.regret_rollouts
                                mean_adversary_penalty = total_adversary_penalty / self.regret_rollouts
                                self.adversary.record_transition(
                                    states=rand_state,
                                    actions=adversary_action,
                                    rewards=(max_adversary_rewards - mean_adversary_rewards - mean_adversary_penalty),
                                    next_states=rand_state,
                                    terminated=torch.ones(terminated.shape, device=self.env.device),
                                    truncated=torch.ones(truncated.shape, device=self.env.device),
                                    infos={},
                                    timestep=(timestep // MAX_EPISODE_LENGTH // self.regret_rollouts),
                                    timesteps=(self.timesteps // MAX_EPISODE_LENGTH // self.regret_rollouts),
                                )
                        else:
                            adversary_rewards = (-1 * episode_rewards) - range_penalty
                            self.adversary.record_transition(
                                states=rand_state,
                                actions=adversary_action,
                                rewards=adversary_rewards,
                                next_states=rand_state,
                                terminated=torch.ones(terminated.shape, device=self.env.device),
                                truncated=torch.ones(truncated.shape, device=self.env.device),
                                infos={},
                                timestep=(timestep // MAX_EPISODE_LENGTH),
                                timesteps=(self.timesteps // MAX_EPISODE_LENGTH),
                            )

                        # log adversary data as necessary
                        if self.log_adversary:
                            adversary_action_log.append(adversary_action.cpu().numpy())
                            adversary_reward_log.append(adversary_rewards.flatten().cpu().numpy())

                    # adversary post interaction
                    if self.positioning_strategy == "regret_adversary":
                        if regret_trials <= 0:
                            self.adversary.post_interaction(
                                timestep=(timestep // MAX_EPISODE_LENGTH // self.regret_rollouts),
                                timesteps=(self.timesteps // MAX_EPISODE_LENGTH // self.regret_rollouts)
                            )

                            max_adversary_rewards = torch.zeros((NUM_ENVS, 1), device=self.env.device)
                            total_adversary_rewards = torch.zeros((NUM_ENVS, 1), device=self.env.device)
                            total_adversary_penalty = torch.zeros((NUM_ENVS, 1), device=self.env.device)
                    else:
                        self.adversary.post_interaction(
                            timestep=(timestep // MAX_EPISODE_LENGTH),
                            timesteps=(self.timesteps // MAX_EPISODE_LENGTH)
                        )

        # dump adversary logs
        if self.log_adversary:
            RESULT_DIR = os.path.join(self.agents.experiment_dir, "adversary_logs")
            os.makedirs(RESULT_DIR, exist_ok=True)

            # dump adversary log to .npy file
            adversary_action_log = np.array(adversary_action_log)
            np.save(os.path.join(RESULT_DIR, "adversary_action_log.npy"), adversary_action_log)
            logger.info("Adversary action log dumped to file")

            # dump adversary reward log to .npy file
            adversary_reward_log = np.array(adversary_reward_log)
            np.save(os.path.join(RESULT_DIR, "adversary_reward_log.npy"), adversary_reward_log)
            logger.info("Adversary reward log dumped to file")


    def single_agent_eval(self) -> None:
        """Evaluate agent

        This method executes the following steps in loop:

        - Compute actions (sequentially)
        - Interact with the environments
        - Render scene
        - Reset environments
        """
        assert self.num_simultaneous_agents == 1, "This method is not allowed for simultaneous agents"
        assert self.env.num_agents == 1, "This method is not allowed for multi-agents"

        # reset env
        states, infos = self.env.reset()

        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar, file=sys.stdout
        ):

            # pre-interaction
            self.agents.pre_interaction(timestep=timestep, timesteps=self.timesteps)

            with torch.no_grad():
                # compute actions
                outputs = self.agents.act(states, timestep=timestep, timesteps=self.timesteps)
                actions = outputs[0] if self.stochastic_evaluation else outputs[-1].get("mean_actions", outputs[0])

                # step the environments
                next_states, rewards, terminated, truncated, infos = self.env.step(actions)

                # render scene
                if not self.headless:
                    self.env.render()

                # write data to TensorBoard
                self.agents.record_transition(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    infos=infos,
                    timestep=timestep,
                    timesteps=self.timesteps,
                )

                # log environment info
                if self.environment_info in infos:
                    for k, v in infos[self.environment_info].items():
                        if isinstance(v, torch.Tensor) and v.numel() == 1:
                            self.agents.track_data(f"Info / {k}", v.item())

            # post-interaction
            super(type(self.agents), self.agents).post_interaction(timestep=timestep, timesteps=self.timesteps)

            # reset environments
            if self.env.num_envs > 1:
                states = next_states
            else:
                if terminated.any() or truncated.any():
                    with torch.no_grad():
                        states, infos = self.env.reset()
                else:
                    states = next_states

    def multi_agent_train(self) -> None:
        """Train multi-agents

        This method executes the following steps in loop:

        - Pre-interaction
        - Compute actions
        - Interact with the environments
        - Render scene
        - Record transitions
        - Post-interaction
        - Reset environments
        """
        assert self.num_simultaneous_agents == 1, "This method is not allowed for simultaneous agents"
        assert self.env.num_agents > 1, "This method is not allowed for single-agent"

        # reset env
        states, infos = self.env.reset()
        shared_states = self.env.state()

        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar, file=sys.stdout
        ):

            # pre-interaction
            self.agents.pre_interaction(timestep=timestep, timesteps=self.timesteps)

            with torch.no_grad():
                # compute actions
                actions = self.agents.act(states, timestep=timestep, timesteps=self.timesteps)[0]

                # step the environments
                next_states, rewards, terminated, truncated, infos = self.env.step(actions)
                shared_next_states = self.env.state()
                infos["shared_states"] = shared_states
                infos["shared_next_states"] = shared_next_states

                # render scene
                if not self.headless:
                    self.env.render()

                # record the environments' transitions
                self.agents.record_transition(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    infos=infos,
                    timestep=timestep,
                    timesteps=self.timesteps,
                )

                # log environment info
                if self.environment_info in infos:
                    for k, v in infos[self.environment_info].items():
                        if isinstance(v, torch.Tensor) and v.numel() == 1:
                            self.agents.track_data(f"Info / {k}", v.item())

            # post-interaction
            self.agents.post_interaction(timestep=timestep, timesteps=self.timesteps)

            # reset environments
            if not self.env.agents:
                with torch.no_grad():
                    states, infos = self.env.reset()
                    shared_states = self.env.state()
            else:
                states = next_states
                shared_states = shared_next_states

    def multi_agent_eval(self) -> None:
        """Evaluate multi-agents

        This method executes the following steps in loop:

        - Compute actions (sequentially)
        - Interact with the environments
        - Render scene
        - Reset environments
        """
        assert self.num_simultaneous_agents == 1, "This method is not allowed for simultaneous agents"
        assert self.env.num_agents > 1, "This method is not allowed for single-agent"

        # reset env
        states, infos = self.env.reset()
        shared_states = self.env.state()

        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar, file=sys.stdout
        ):

            # pre-interaction
            self.agents.pre_interaction(timestep=timestep, timesteps=self.timesteps)

            with torch.no_grad():
                # compute actions
                outputs = self.agents.act(states, timestep=timestep, timesteps=self.timesteps)
                actions = (
                    outputs[0]
                    if self.stochastic_evaluation
                    else {k: outputs[-1][k].get("mean_actions", outputs[0][k]) for k in outputs[-1]}
                )

                # step the environments
                next_states, rewards, terminated, truncated, infos = self.env.step(actions)
                shared_next_states = self.env.state()
                infos["shared_states"] = shared_states
                infos["shared_next_states"] = shared_next_states

                # render scene
                if not self.headless:
                    self.env.render()

                # write data to TensorBoard
                self.agents.record_transition(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    infos=infos,
                    timestep=timestep,
                    timesteps=self.timesteps,
                )

                # log environment info
                if self.environment_info in infos:
                    for k, v in infos[self.environment_info].items():
                        if isinstance(v, torch.Tensor) and v.numel() == 1:
                            self.agents.track_data(f"Info / {k}", v.item())

            # post-interaction
            super(type(self.agents), self.agents).post_interaction(timestep=timestep, timesteps=self.timesteps)

            # reset environments
            if not self.env.agents:
                with torch.no_grad():
                    states, infos = self.env.reset()
                    shared_states = self.env.state()
            else:
                states = next_states
                shared_states = shared_next_states

    def _isaaclab_env(self) -> Wrapper:
        """Get the Isaac Lab environment through all the wrappers

        :return: Environment
        :rtype: AdversarialManagerBasedRLEnv
        """
        res = self.env._env.env
        if hasattr(res, "env"):
            res = res.env
        return res
