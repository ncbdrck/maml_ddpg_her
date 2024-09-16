import torch
import os
from datetime import datetime
import numpy as np
from mpi4py import MPI

from mpi_utils.mpi_utils import sync_networks, sync_grads
from rl_modules.replay_buffer import replay_buffer
from rl_modules.vanilla.models import actor, critic
from mpi_utils.normalizer import normalizer
from her_modules.her import her_sampler

# additional imports
from torch.utils.tensorboard import SummaryWriter
import wandb
import time
import copy

"""
MAML 
- ddpg with HER (MPI-version)
- added loading global step for the outer loop
- does not use deep copy to initialize the target networks
- Removed sync_networks for inner networks since we are initializing them with the main networks
- clip the action again in case of adding random actions
- seperated meta and inner loss calculation
- now we sample all the tasks first then calculate the meta loss for each task
- added n_meta_batches to update the main model in the outer loop instead of using the same number of batches as the inner loop
"""

class maml_ddpg_her_agent:
    def __init__(self, args, envs, env_params_list, env_names, seed):
        # Meta-parameters for MAML
        self.alpha = args.maml_alpha
        self.beta = args.maml_beta

        # uncomment the following line to detect the anomaly - debugging
        # torch.autograd.set_detect_anomaly(True)

        # Logging variables
        self.success_window = args.log_interval
        self.ep_len_window = args.log_interval
        self.reward_window = args.log_interval
        self.log_interval = args.log_interval  # todo: only use this later

        # Individual environment logging
        self.success_history_env = [[] for _ in range(len(envs))]
        self.ep_len_history_env = [[] for _ in range(len(envs))]
        self.reward_history_env = [[] for _ in range(len(envs))]
        self.global_step_env = [0 for _ in range(len(envs))]
        self.global_step = 0  # todo: for outer loop?
        self.meta_update_counter = 0  # counter for the meta updates

        # for Saving and loading parameters
        self.exp_name = os.path.basename(__file__)[: -len(".py")]
        self.run_name = f"{args.exp_name}__{self.exp_name}__{args.seed}__{int(time.time())}"
        self.rank = MPI.COMM_WORLD.Get_rank()

        # todo: Initialize Weights & Biases (not checked)
        if args.track and self.rank == 0:
            wandb.login()  # for offline mode, remove this line
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=vars(args),
                name=self.run_name,
                monitor_gym=True,
                save_code=True,
            )

        # Initialize Tensorboard Summary Writer
        if self.rank == 0:
            self.writer = SummaryWriter(f"runs/{self.run_name}")
            self.writer.add_text(
                "hyperparameters",
                "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
            )

        self.args = args
        self.envs = envs
        # we can comment out the following line to see what we need to change for multiple environments with different
        # parameters
        self.env_params = env_params_list[0]  # todo: for now, we assume all environments have the same parameters
        self.env_params_list = env_params_list  # todo: if we want to use envs with different env parameters
        self.env_names = env_names
        self._seed = seed

        # Create the networks (outer loop)
        self.actor_network = actor(self.env_params)  # todo: need to fix it later since we do max action normalization?
        self.critic_network = critic(self.env_params)
        self.actor_target_network = actor(self.env_params)
        self.critic_target_network = critic(self.env_params)

        # Load the model if specified
        global_step = 0  # initialize the global step to 0
        if self.args.load_model and self.rank == 0:
            load_path = f"runs/{args.load_run_name}/{args.load_model_name}"
            if os.path.exists(load_path):
                print(f"\033[92mLoading model from {load_path}\033[0m")
                checkpoint = torch.load(load_path)
                self.actor_network.load_state_dict(checkpoint['actor_state_dict'])
                self.critic_network.load_state_dict(checkpoint['critic_state_dict'])
                self.actor_target_network.load_state_dict(checkpoint['actor_target_state_dict'])
                self.critic_target_network.load_state_dict(checkpoint['critic_target_state_dict'])

                # load the global step
                if self.args.continue_training_log:
                    global_step = checkpoint['global_step']

                # todo: we are not finishing yet - Jay
                # todo: load the parameters related to the global steps for each environment
                # todo; load the parameters related to the replay buffers and normalizers (probably not needed)

            else:
                raise FileNotFoundError(f"Model not found at {load_path}")

        # load the global step if only we are continuing the training
        if self.args.continue_training_log and self.args.load_model:

            # broadcast the global step to all the cpus
            global_step = MPI.COMM_WORLD.bcast(global_step, root=0)

            # Set the local global step
            if global_step > 0:
                self.global_step = global_step

        # Sync the networks across CPUs
        sync_networks(self.actor_network)
        sync_networks(self.critic_network)

        # Print the number of trainable parameters
        if self.rank == 0:
            actor_parm_cnt = self.count_trainable_parameters(self.actor_network)
            critic_parm_cnt = self.count_trainable_parameters(self.critic_network)
            print(f"\033[94mActor trainable parameters: {actor_parm_cnt:,}\033[0m")
            print(f"\033[94mCritic trainable parameters: {critic_parm_cnt:,}\033[0m")

        # Sync the target networks
        if self.args.load_model:
            # sync the target networks
            sync_networks(self.actor_target_network)
            sync_networks(self.critic_target_network)
        else:
            # load the weights into the target networks
            self.actor_target_network.load_state_dict(self.actor_network.state_dict())
            self.critic_target_network.load_state_dict(self.critic_network.state_dict())

        # Use GPU if available
        if self.args.cuda:
            self.actor_network.cuda()
            self.critic_network.cuda()
            self.actor_target_network.cuda()
            self.critic_target_network.cuda()

        # Create the optimizer for outer loop
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.beta)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.beta)

        # create actor and critic networks for each environment - for inner loop
        self.inner_actor_networks = [actor(env_params) for env_params in env_params_list]
        self.inner_critic_networks = [critic(env_params) for env_params in env_params_list]
        self.inner_actor_target_networks = [actor(env_params) for env_params in env_params_list]
        self.inner_critic_target_networks = [critic(env_params) for env_params in env_params_list]

        # initialize the inner networks and sync them across CPUs
        for inner_actor, inner_actor_target in zip(self.inner_actor_networks, self.inner_actor_target_networks):
            inner_actor.load_state_dict(self.actor_network.state_dict())
            inner_actor_target.load_state_dict(self.actor_target_network.state_dict())

        # copy critic parameters
        for inner_critic, inner_critic_target in zip(self.inner_critic_networks, self.inner_critic_target_networks):
            inner_critic.load_state_dict(self.critic_network.state_dict())
            inner_critic_target.load_state_dict(self.critic_target_network.state_dict())

        # Use GPU if available, send the inner networks to the GPU
        if self.args.cuda:
            for inner_actor, inner_critic, inner_actor_target, inner_critic_target in zip(self.inner_actor_networks,
                                                                                          self.inner_critic_networks,
                                                                                          self.inner_actor_target_networks,
                                                                                          self.inner_critic_target_networks):
                inner_actor.cuda()
                inner_critic.cuda()
                inner_actor_target.cuda()
                inner_critic_target.cuda()


        # create optimizers for each environment
        self.inner_actor_optims = [torch.optim.Adam(actor_network.parameters(), lr=self.alpha) for actor_network in
                                   self.inner_actor_networks]
        self.inner_critic_optims = [torch.optim.Adam(critic_network.parameters(), lr=self.alpha) for critic_network in
                                    self.inner_critic_networks]

        # Create the HER modules
        self.her_modules = [her_sampler(self.args.replay_strategy, self.args.replay_k, env.compute_reward) for env in
                            envs]

        # Create the replay buffers for inner loop
        # self.inner_buffers = [replay_buffer(self.env_params, self.args.buffer_size, her_module.sample_her_transitions)
        #                       for her_module in self.her_modules]
        # to make it more general, we can use the following
        self.inner_buffers = [replay_buffer(env_params, self.args.buffer_size, her_module.sample_her_transitions)
                              for env_params, her_module in zip(env_params_list, self.her_modules)]

        # Create the replay buffers for outer loop - meta updates
        # self.meta_buffers = [replay_buffer(self.env_params, self.args.buffer_size, her_module.sample_her_transitions)
        #                      for her_module in self.her_modules]
        # to make it more general, we can use the following
        self.meta_buffers = [replay_buffer(env_params, self.args.buffer_size, her_module.sample_her_transitions)
                                for env_params, her_module in zip(env_params_list, self.her_modules)]
        # todo: ideally, the meta replay buffers doesn't need to have this much capacity since we empty it after every
        # outer loop as we only need to update parameters using experiences after the inner loop updates
        # So we can use a size with args.maml_K * env_params['max_timesteps'] * args.n_batches
        # self.meta_buffers = [replay_buffer(self.env_params,
        #                                    self.args.maml_K * self.env_params['max_timesteps'] * self.args.n_batches,
        #                                    her_module.sample_her_transitions) for her_module in self.her_modules]


        # Create the normalizers (one normalizer for all the environments) - not a good idea
        # self.o_norms = normalizer(size=self.env_params['obs'], default_clip_range=self.args.clip_range)
        # self.g_norms = normalizer(size=self.env_params['goal'], default_clip_range=self.args.clip_range)

        # Create the normalizers for each environment
        # better to use separate normalizers for each env since the observations and goal can be different
        # even though obs space and the goal space are the same, the values can be different
        self.o_norms_list = [normalizer(size=env_params['obs'], default_clip_range=self.args.clip_range) for env_params
                             in env_params_list]
        self.g_norms_list = [normalizer(size=env_params['goal'], default_clip_range=self.args.clip_range) for env_params
                             in env_params_list]

        # Create directory to store the model
        # todo: this is not needed for now
        if self.rank == 0:
            if not os.path.exists(self.args.save_dir):
                os.mkdir(self.args.save_dir)
            self.model_path = os.path.join(self.args.save_dir, self.args.exp_name, self.run_name)
            # create the directory if it doesn't exist
            if not os.path.exists(self.model_path):
                os.makedirs(self.model_path)
        self.model_path_maml =  f"runs/{self.run_name}/{self.exp_name}"

    def learn(self):
        """
        Train the agent. This is the main method that contains the training loop
        """
        for epoch in range(self.args.n_epochs):
            for cycle in range(self.args.n_cycles):
                tasks = self.sample_tasks()
                self.outer_loop(tasks)

                # Evaluate the agent after every cycle
                self.evaluation_and_logging_cycle(cycle, epoch)

            # # Evaluate the agent after every epoch
            # success_rate, reward, ep_len = self.evaluate_agent()
            # if MPI.COMM_WORLD.Get_rank() == 0:
            #     print(f'[{datetime.now()}] epoch: {epoch + 1}, eval success rate: {success_rate:.3f}')
            #     self.writer.add_scalar("rollouts/eval_success_rate", success_rate, epoch)
            #     self.writer.add_scalar("rollouts/eval_reward", reward, epoch)
            #     self.writer.add_scalar("rollouts/eval_ep_len", ep_len, epoch)
            #     if self.args.track:
            #         wandb.log({"rollouts/eval_success_rate": success_rate}, step=epoch)
            #         wandb.log({"rollouts/eval_reward": reward}, step=epoch)
            #         wandb.log({"rollouts/eval_ep_len": ep_len}, step=epoch)

        # save the main model
        if self.rank == 0 and self.args.save_model:

            print("\033[92m" + f"Saving the model at {self.model_path_maml}" + "\033[0m")
            torch.save({
                'actor_state_dict': self.actor_network.state_dict(),
                'critic_state_dict': self.critic_network.state_dict(),
                'actor_target_state_dict': self.actor_target_network.state_dict(),
                'critic_target_state_dict': self.critic_target_network.state_dict(),
            }, self.model_path_maml)

    def sample_tasks(self):
        """
        Sample a list of tasks for the meta-update
        :return: return a list of tasks
        """
        return np.random.randint(0, len(self.envs), size=self.args.maml_num_tasks).tolist()

    def outer_loop(self, tasks):
        # total_critic_loss = 0
        # total_actor_loss = 0

        # increase the global step
        self.global_step += 1

        for env_idx in tasks:
            # update the inner networks
            self.inner_loop(env_idx)

        # todo: do we need to do this (number of batches) times?
        for _ in range(self.args.n_meta_batches):
            total_actor_loss = torch.tensor(0.0, dtype=torch.float32)
            total_critic_loss = torch.tensor(0.0, dtype=torch.float32)

            if self.args.cuda:
                total_actor_loss = total_actor_loss.cuda()
                total_critic_loss = total_critic_loss.cuda()

            # calculate the meta loss for each environment
            for env_idx in tasks:
                meta_actor_loss, meta_critic_loss = self.compute_loss(env_idx, meta=True)
                if self.args.debug:
                    # todo: debug - print the type of the data -  we need tensor
                    print(f"meta_actor_loss: {type(meta_actor_loss)}, meta_critic_loss: {type(meta_critic_loss)}")
                    # print meta_actor_loss so we can check if it has gradient tracking
                    print(f"meta_actor_loss: {meta_actor_loss.requires_grad}")
                    print(f"meta_critic_loss: {meta_critic_loss.requires_grad}")

                # Accumulate losses
                total_actor_loss += meta_actor_loss
                total_critic_loss += meta_critic_loss

            # Average the losses across tasks
            total_actor_loss /= len(tasks)
            total_critic_loss /= len(tasks)
            # todo; debug
            if self.args.debug:
                print(f"total_critic_loss: {total_critic_loss}, total_actor_loss: {total_actor_loss}")

            # log the losses for meta updates
            self.meta_update_counter += 1
            if self.rank == 0 and self.meta_update_counter % 20 == 0:
                self.writer.add_scalar("rollout/meta_actor_loss", total_actor_loss, self.meta_update_counter)
                self.writer.add_scalar("rollout/meta_critic_loss", total_critic_loss, self.meta_update_counter)
                if self.args.track:
                    wandb.log({"rollout/meta_actor_loss": total_actor_loss}, step=self.meta_update_counter)
                    wandb.log({"rollout/meta_critic_loss": total_critic_loss}, step=self.meta_update_counter)

            self.actor_optim.zero_grad()
            total_actor_loss.backward()
            sync_grads(self.actor_network)
            self.actor_optim.step()

            self.critic_optim.zero_grad()
            total_critic_loss.backward()
            sync_grads(self.critic_network)
            self.critic_optim.step()

        # soft update the target networks
        self._soft_update_target_network(self.critic_target_network, self.critic_network)
        self._soft_update_target_network(self.actor_target_network, self.actor_network)

    def evaluation_and_logging_cycle(self, step, epoch):
        """
        Evaluate the agent and log the results

        :param epoch: The current epoch
        :param step: The current step
        :return:
        """

        success_rate, reward, ep_len = self.evaluate_agent()
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(f'[{datetime.now()}] Epoch: {epoch+1}, Cycle : {step + 1}, eval success rate: {success_rate:.3f}')
            self.writer.add_scalar("Cycle/eval_success_rate", success_rate, self.global_step)
            self.writer.add_scalar("Cycle/eval_reward", reward, self.global_step)
            self.writer.add_scalar("Cycle/eval_ep_len", ep_len, self.global_step)
            if self.args.track:
                wandb.log({"Cycle/eval_success_rate": success_rate}, step=self.global_step)
                wandb.log({"Cycle/eval_reward": reward}, step=self.global_step)
                wandb.log({"Cycle/eval_ep_len": ep_len}, step=self.global_step)

    # soft update
    def _soft_update_target_network(self, target, source):
        """
        Soft update the target network
        :param target: target network
        :param source: source network
        :return:
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    def compute_loss(self, env_idx, meta=False):
        """
        Compute the actor and critic loss
        :param env_idx: Index of the environment
        :param meta: To indicate if the loss is for meta updates
        :return: actor_loss, critic_loss
        """

        # sample the episodes
        if meta:
            transitions = self.meta_buffers[env_idx].sample(self.args.batch_size)
        else:
            transitions = self.inner_buffers[env_idx].sample(self.args.batch_size)

        # Pre-process the observation and goal
        o, o_next, g = transitions['obs'], transitions['obs_next'], transitions['g']
        transitions['obs'], transitions['g'] = self._preproc_og(o, g)
        transitions['obs_next'], transitions['g_next'] = self._preproc_og(o_next, g)

        # Normalize the inputs
        obs_norm = self.o_norms_list[env_idx].normalize(transitions['obs'])
        g_norm = self.g_norms_list[env_idx].normalize(transitions['g'])
        inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)

        obs_next_norm = self.o_norms_list[env_idx].normalize(transitions['obs_next'])
        g_next_norm = self.g_norms_list[env_idx].normalize(transitions['g_next'])
        inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm], axis=1)

        # Convert to tensors
        inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
        inputs_next_norm_tensor = torch.tensor(inputs_next_norm, dtype=torch.float32)
        actions_tensor = torch.tensor(transitions['actions'], dtype=torch.float32)
        r_tensor = torch.tensor(transitions['r'], dtype=torch.float32)

        if self.args.cuda:
            inputs_norm_tensor = inputs_norm_tensor.cuda()
            inputs_next_norm_tensor = inputs_next_norm_tensor.cuda()
            actions_tensor = actions_tensor.cuda()
            r_tensor = r_tensor.cuda()

        # Compute the target Q-values
        with torch.no_grad():
            if meta:
                actions_next = self.actor_target_network(inputs_next_norm_tensor)
                q_next_value = self.critic_target_network(inputs_next_norm_tensor, actions_next)
                q_next_value = q_next_value.detach()
                target_q_value = r_tensor + self.args.gamma * q_next_value
                target_q_value = target_q_value.detach()
                clip_return = 1 / (1 - self.args.gamma)
                target_q_value = torch.clamp(target_q_value, -clip_return, 0)
            else:
                actions_next = self.inner_actor_target_networks[env_idx](inputs_next_norm_tensor)
                q_next_value = self.inner_critic_target_networks[env_idx](inputs_next_norm_tensor, actions_next)
                q_next_value = q_next_value.detach()
                target_q_value = r_tensor + self.args.gamma * q_next_value
                target_q_value = target_q_value.detach()
                clip_return = 1 / (1 - self.args.gamma)
                target_q_value = torch.clamp(target_q_value, -clip_return, 0)

        # Compute current Q-values and the critic loss
        if meta:
            real_q_value = self.critic_network(inputs_norm_tensor, actions_tensor)
            critic_loss = (target_q_value - real_q_value).pow(2).mean()
        else:
            real_q_value = self.inner_critic_networks[env_idx](inputs_norm_tensor, actions_tensor)
            critic_loss = (target_q_value - real_q_value).pow(2).mean()

        # Compute critic loss
        # critic_loss = torch.nn.functional.mse_loss(real_q_value, target_q_value)
        # critic_loss = (target_q_value - real_q_value).pow(2).mean()

        # Compute actor loss
        if meta:
            actions_real = self.actor_network(inputs_norm_tensor)
            actor_loss = -self.critic_network(inputs_norm_tensor, actions_real).mean()
            actor_loss += self.args.action_l2 * (actions_real / self.env_params_list[env_idx]['action_max']).pow(
                2).mean()
        else:
            actions_real = self.inner_actor_networks[env_idx](inputs_norm_tensor)
            actor_loss = -self.inner_critic_networks[env_idx](inputs_norm_tensor, actions_real).mean()
            actor_loss += self.args.action_l2 * (actions_real / self.env_params_list[env_idx]['action_max']).pow(
                2).mean()

        # Add action regularization to keep the actions in check
        # actor_loss += self.args.action_l2 * (actions_real / self.env_params_list[env_idx]['action_max']).pow(2).mean()

        # log the losses for inner loop updates
        if self.rank ==0 and not meta:
            self.writer.add_scalar(f"env_{env_idx}/actor_loss", actor_loss, self.global_step_env[env_idx])
            self.writer.add_scalar(f"env_{env_idx}/critic_loss", critic_loss, self.global_step_env[env_idx])
            if self.args.track:
                wandb.log({f"env_{env_idx}/actor_loss": actor_loss}, step=self.global_step_env[env_idx])
                wandb.log({f"env_{env_idx}/critic_loss": critic_loss}, step=self.global_step_env[env_idx])

        return actor_loss, critic_loss

    def inner_loop(self, env_idx):

        # retrieve the task-specific variables
        env = self.envs[env_idx]

        # copy the main networks to the task-specific networks
        self.inner_actor_networks[env_idx].load_state_dict(self.actor_network.state_dict())
        self.inner_critic_networks[env_idx].load_state_dict(self.critic_network.state_dict())
        self.inner_actor_target_networks[env_idx].load_state_dict(self.actor_target_network.state_dict())
        self.inner_critic_target_networks[env_idx].load_state_dict(self.critic_target_network.state_dict())

        # Sample K trajectories
        mb_obs, mb_ag, mb_g, mb_actions = [], [], [], []
        for _ in range(self.args.maml_K):
            ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
            observation, _ = env.reset(seed=self._seed)
            obs = observation['observation']
            ag = observation['achieved_goal']
            g = observation['desired_goal']
            ep_reward = 0
            ep_done = False

            # todo: we can use the commented line below to use the max_timesteps for each environment
            # for t in range(self.env_params_list[env_idx]['max_timesteps']):
            for t in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    input_tensor = self._preproc_inputs(obs, g, env_idx)
                    pi = self.inner_actor_networks[env_idx](input_tensor)
                    action = self._select_actions(pi, env_idx)
                observation_new, r, term, trunc, info = env.step(action)
                obs_new = observation_new['observation']
                ag_new = observation_new['achieved_goal']

                # store the episode
                ep_obs.append(obs.copy())
                ep_ag.append(ag.copy())
                ep_g.append(g.copy())
                ep_actions.append(action.copy())

                # update the variables
                obs = obs_new
                ag = ag_new

                # for logging
                self.global_step_env[env_idx] += 1
                ep_reward += r

                # check if the episode is done
                if term or trunc or t + 1 == self.env_params['max_timesteps'] and not ep_done:
                    ep_done = True

                    # log the episode
                    if self.rank == 0:
                        self._log_episode(env_idx, t + 1, ep_reward, info.get('is_success', 0))

            ep_obs.append(obs.copy())
            ep_ag.append(ag.copy())
            mb_obs.append(ep_obs)
            mb_ag.append(ep_ag)
            mb_g.append(ep_g)
            mb_actions.append(ep_actions)

        # convert them into np arrays
        mb_obs = np.array(mb_obs)
        mb_ag = np.array(mb_ag)
        mb_g = np.array(mb_g)
        mb_actions = np.array(mb_actions)

        # store the episodes
        self.inner_buffers[env_idx].store_episode([mb_obs, mb_ag, mb_g, mb_actions])
        self._update_normalizer([mb_obs, mb_ag, mb_g, mb_actions], env_idx)

        # train the task-specific networks
        for _ in range(self.args.n_batches):
            inner_actor_loss, inner_critic_loss = self.compute_loss(env_idx)

            self.inner_actor_optims[env_idx].zero_grad()
            inner_actor_loss.backward()
            # todo: Do we need to sync the gradients across the CPUs?
            # sync_grads(self.inner_actor_networks[env_idx])
            self.inner_actor_optims[env_idx].step()

            self.inner_critic_optims[env_idx].zero_grad()
            inner_critic_loss.backward()
            # todo: Same here?
            # sync_grads(self.inner_critic_networks[env_idx])
            self.inner_critic_optims[env_idx].step()

        # soft update the target networks
        # we need this since we are sampling new trajectories using the updated task-specific networks
        self._soft_update_target_network(self.inner_actor_target_networks[env_idx], self.inner_actor_networks[env_idx])
        self._soft_update_target_network(self.inner_critic_target_networks[env_idx], self.inner_critic_networks[env_idx])


        # Sample new trajectories using updated task-specific networks
        mb_obs, mb_ag, mb_g, mb_actions = [], [], [], []
        for _ in range(self.args.maml_K):
            ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
            observation, _ = env.reset(seed=self._seed)
            obs = observation['observation']
            ag = observation['achieved_goal']
            g = observation['desired_goal']
            ep_reward = 0
            ep_done = False

            # todo: we can use the commented line below to use the max_timesteps for each environment
            # for t in range(self.env_params_list[env_idx]['max_timesteps']):
            for t in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    input_tensor = self._preproc_inputs(obs, g, env_idx)
                    pi = self.inner_actor_networks[env_idx](input_tensor)
                    action = self._select_actions(pi, env_idx)
                observation_new, r, term, trunc, info = env.step(action)
                obs_new = observation_new['observation']
                ag_new = observation_new['achieved_goal']

                ep_obs.append(obs.copy())
                ep_ag.append(ag.copy())
                ep_g.append(g.copy())
                ep_actions.append(action.copy())
                obs = obs_new
                ag = ag_new
                ep_reward += r
                self.global_step_env[env_idx] += 1

                if term or trunc or t + 1 == self.env_params['max_timesteps'] and not ep_done:
                    ep_done = True

                    # log the episode
                    if self.rank == 0:
                        self._log_episode(env_idx, t + 1, ep_reward, info.get('is_success', 0))

            ep_obs.append(obs.copy())
            ep_ag.append(ag.copy())
            mb_obs.append(ep_obs)
            mb_ag.append(ep_ag)
            mb_g.append(ep_g)
            mb_actions.append(ep_actions)

        mb_obs = np.array(mb_obs)
        mb_ag = np.array(mb_ag)
        mb_g = np.array(mb_g)
        mb_actions = np.array(mb_actions)

        # clear and store the episodes in the meta replay buffer
        self.meta_buffers[env_idx].clear_buffer()
        self.meta_buffers[env_idx].store_episode([mb_obs, mb_ag, mb_g, mb_actions])
        self._update_normalizer([mb_obs, mb_ag, mb_g, mb_actions], env_idx)


    def _log_episode(self, env_idx, episode_length, episode_reward, is_success):
        """
        Log the episode statistics

        :param env_idx:
        :param episode_length:
        :param episode_reward:
        :param is_success:
        :return:
        """

        # get env name
        env_name = self.env_names[env_idx]

        # append the episode length and calculate the mean
        self.ep_len_history_env[env_idx].append(episode_length)
        if len(self.ep_len_history_env[env_idx]) >= self.ep_len_window:
            mean_ep_len = np.mean(self.ep_len_history_env[env_idx][-self.ep_len_window:])
            self.ep_len_history_env[env_idx] = self.ep_len_history_env[env_idx][-self.ep_len_window:]
        else:
            mean_ep_len = np.mean(self.ep_len_history_env[env_idx])
        # log the mean episode length
        if self.rank == 0:
            self.writer.add_scalar(f"{env_name}/mean_ep_length", mean_ep_len, self.global_step_env[env_idx])
            if self.args.track:
                wandb.log({f"{env_name}/mean_ep_length": mean_ep_len}, step=self.global_step_env[env_idx])

        # append the episode reward and calculate the mean
        self.reward_history_env[env_idx].append(episode_reward)
        if len(self.reward_history_env[env_idx]) >= self.reward_window:
            mean_reward = np.mean(self.reward_history_env[env_idx][-self.reward_window:])
            self.reward_history_env[env_idx] = self.reward_history_env[env_idx][-self.reward_window:]
        else:
            mean_reward = np.mean(self.reward_history_env[env_idx])
        # log the mean episode reward
        if self.rank == 0:
            self.writer.add_scalar(f"{env_name}/mean_reward", mean_reward, self.global_step_env[env_idx])
            if self.args.track:
                wandb.log({f"{env_name}/mean_reward": mean_reward}, step=self.global_step_env[env_idx])

        # append the success rate and calculate the mean
        self.success_history_env[env_idx].append(is_success)
        if len(self.success_history_env[env_idx]) >= self.success_window:
            mean_success_rate = np.mean(self.success_history_env[env_idx][-self.success_window:])
            self.success_history_env[env_idx] = self.success_history_env[env_idx][-self.success_window:]
        else:
            mean_success_rate = np.mean(self.success_history_env[env_idx])
        # log the mean success rate
        if self.rank == 0:
            self.writer.add_scalar(f"{env_name}/mean_success_rate", mean_success_rate,
                                   self.global_step_env[env_idx])
            if self.args.track:
                wandb.log({f"{env_name}/mean_success_rate": mean_success_rate}, step=self.global_step_env[env_idx])

    def _select_actions(self, pi, env_idx):
        action = pi.cpu().numpy().squeeze()
        # add the gaussian
        action += self.args.noise_eps * self.env_params_list[env_idx]['action_max'] * np.random.randn(*action.shape)
        action = np.clip(action, -self.env_params_list[env_idx]['action_max'], self.env_params_list[env_idx]['action_max'])
        # random actions...
        random_actions = np.random.uniform(low=-self.env_params_list[env_idx]['action_max'],
                                           high=self.env_params_list[env_idx]['action_max'],
                                           size=self.env_params_list[env_idx]['action'])
        # choose if you use the random actions
        action += np.random.binomial(1, self.args.random_eps, 1)[0] * (random_actions - action)
        # clip the actions again in case of the random component
        action = np.clip(action, -self.env_params_list[env_idx]['action_max'], self.env_params_list[env_idx]['action_max'])
        return action



    def _preproc_inputs(self, obs, g, env_idx):
        """
        Preprocess the inputs for the networks

        :param obs:
        :param g:
        :param env_idx:
        :return: inputs_tensor for the networks
        """

        # retrieve the normalizers
        o_norm = self.o_norms_list[env_idx]
        g_norm = self.g_norms_list[env_idx]

        obs_norm = o_norm.normalize(obs)
        g_norm = g_norm.normalize(g)
        inputs = np.concatenate([obs_norm, g_norm])
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs_tensor = inputs_tensor.cuda()
        return inputs_tensor

    def _update_normalizer(self, episode_batch, env_idx):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        mb_obs_next = mb_obs[:, 1:, :]
        mb_ag_next = mb_ag[:, 1:, :]
        num_transitions = mb_actions.shape[1]
        buffer_temp = {'obs': mb_obs,
                       'ag': mb_ag,
                       'g': mb_g,
                       'actions': mb_actions,
                       'obs_next': mb_obs_next,
                       'ag_next': mb_ag_next,
                       }
        her_module = self.her_modules[env_idx]
        transitions = her_module.sample_her_transitions(buffer_temp, num_transitions)
        obs, g = transitions['obs'], transitions['g']
        # pre-process the obs and g
        transitions['obs'], transitions['g'] = self._preproc_og(obs, g)
        # update
        self.o_norms_list[env_idx].update(transitions['obs'])
        self.g_norms_list[env_idx].update(transitions['g'])
        # recompute the stats
        self.o_norms_list[env_idx].recompute_stats()
        self.g_norms_list[env_idx].recompute_stats()

    def _preproc_og(self, o, g):
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        return o, g

    @staticmethod
    def count_trainable_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def evaluate_agent(self):
        total_success_rate = []
        total_ep_len = []
        total_reward = []

        for env_name, env, o_norm, g_norm in zip(self.env_names, self.envs, self.o_norms_list, self.g_norms_list):
            env_success_rate = []
            env_ep_len = []
            env_reward = []

            # get the environment index
            env_idx = self.get_env_idx(env_name)

            for _ in range(self.args.n_test_rollouts):
                observation, _ = env.reset()
                obs = observation['observation']
                g = observation['desired_goal']
                current_ep_len = 0
                current_ep_reward = 0
                done_flag = False

                for t in range(self.env_params['max_timesteps']):
                    with torch.no_grad():
                        input_tensor = self._preproc_inputs(obs, g, env_idx)
                        pi = self.actor_network(input_tensor)
                        actions = pi.detach().cpu().numpy().squeeze()
                    observation_new, reward, term, trunc, info = env.step(actions)

                    if not done_flag:
                        current_ep_len += 1
                        current_ep_reward += reward

                        if term or trunc or t + 1 == self.env_params['max_timesteps']:
                            done_flag = True
                            env_success_rate.append(info['is_success'])

                    obs = observation_new['observation']
                    g = observation_new['desired_goal']

                env_ep_len.append(current_ep_len)
                env_reward.append(current_ep_reward)

            total_success_rate.append(np.mean(env_success_rate))
            total_ep_len.append(np.mean(env_ep_len))
            total_reward.append(np.mean(env_reward))

        local_success_rate = np.mean(total_success_rate)
        global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
        local_ep_len = np.mean(total_ep_len)
        global_ep_len = MPI.COMM_WORLD.allreduce(local_ep_len, op=MPI.SUM)
        local_reward = np.mean(total_reward)
        global_reward = MPI.COMM_WORLD.allreduce(local_reward, op=MPI.SUM)

        success_rate = global_success_rate / MPI.COMM_WORLD.Get_size()
        ep_len = global_ep_len / MPI.COMM_WORLD.Get_size()
        reward = global_reward / MPI.COMM_WORLD.Get_size()

        return success_rate, reward, ep_len

    def save_model(self, env_name: str):
        """
        This lets us save the model for each environment when training multiple environments

        :param env_name:
        :return:
        """
        env_idx = self.get_env_idx(env_name)
        torch.save({
            'env_name': env_name,
            'o_norm_mean': self.o_norms_list[env_idx].mean,
            'o_norm_std': self.o_norms_list[env_idx].std,
            'g_norm_mean': self.g_norms_list[env_idx].mean,
            'g_norm_std': self.g_norms_list[env_idx].std,
        }, f"{self.model_path}/norm_{env_name}.pt")

        torch.save({
            'env_name': env_name,
            'actor_state_dict': self.actor_network.state_dict(),
            'critic_state_dict': self.critic_network.state_dict(),
            'actor_target_state_dict': self.actor_target_network.state_dict(),
            'critic_target_state_dict': self.critic_target_network.state_dict(),
        }, f"{self.model_path}/model_{env_name}.pt")

    def get_env_idx(self, env_name: str):
        """
        get the index of the environment in the list of environments

        :param env_name:
        :return:
        """
        return self.env_names.index(env_name)

    def get_env_name(self, env_idx: int):
        """
        get the name of the environment from the index

        :param env_idx:
        :return:
        """
        return self.env_names[env_idx]

