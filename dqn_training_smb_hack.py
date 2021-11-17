import gc
from itertools import count
import os
import random
import time

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import smb

def get_make_env_fn(**kargs):
    def make_env_fn(env_name, seed=None, render=None, record=False,
                    unwrapped=False, monitor_mode=None, 
                    inner_wrappers=None, outer_wrappers=None):
        env = gym.make(env_name)
        env.seed(42)
        return env
    return make_env_fn, kargs

class GreedyStrategy:
    def __init__(self):
        super().__init__()
        self.exploratory_action_taken = False

    def select_action(self, model: nn.Module, state: np.ndarray) -> int:
        with torch.no_grad():
            q_values = model(state).cpu().detach().data.numpy().squeeze()
            return np.argmax(q_values)

class EGreedyStrategy:
    def __init__(self, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon
        self.exploratory_action_taken = None

    def select_action(self, model: nn.Module, state: np.ndarray) -> int:
        self.exploratory_action_taken = False
        with torch.no_grad():
            q_values = model(state).cpu().detach().data.numpy().squeeze()

        if np.random.rand() > self.epsilon:
            action = np.argmax(q_values)
        else: 
            action = np.random.randint(len(q_values))

        self.exploratory_action_taken = action != np.argmax(q_values)
        return action


class FCQ(nn.Module):
    def __init__(self, 
                 input_dim, 
                 output_dim, 
                 hidden_dims=(32,32), 
                 activation_fc=F.relu):
        super(FCQ, self).__init__()
        self.activation_fc = activation_fc

        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
        self.device = torch.device(device)
        self.to(self.device)
        
    def _format(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, 
                                device=self.device, 
                                dtype=torch.float32)
            x = x.unsqueeze(0)
        return x

    def forward(self, state):
        x = self._format(state)
        x = self.activation_fc(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))
        x = self.output_layer(x)
        return x
    
    def numpy_float_to_device(self, variable):
        variable = torch.from_numpy(variable).float().to(self.device)
        return variable
    
    def load(self, experiences):
        states, actions, new_states, rewards, is_terminals = experiences
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        new_states = torch.from_numpy(new_states).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        is_terminals = torch.from_numpy(is_terminals).float().to(self.device)
        return states, actions, new_states, rewards, is_terminals

class ReplayBuffer:
    def __init__(self,
                max_size: int=10000,
                batch_size: int=64):
        self.ss_mem = np.empty((max_size), dtype=np.ndarray)
        self.as_mem = np.empty((max_size), dtype=np.ndarray)
        self.rs_mem = np.empty((max_size), dtype=np.ndarray)
        self.ps_mem = np.empty((max_size), dtype=np.ndarray)
        self.ds_mem = np.empty((max_size), dtype=np.ndarray)
        
        self.max_size = max_size
        self.batch_size = batch_size
        self._idx: int = 0
        self.size: int = 0


    def store(self, sample) -> None:
        """
        Args:
            sample: (state, action, reward, new_state, is_failure)
        """
        s, a, r, p, d = sample
        self.ss_mem[self._idx] = s
        self.as_mem[self._idx] = a
        self.rs_mem[self._idx] = r
        self.ps_mem[self._idx] = p
        self.ds_mem[self._idx] = d

        self._idx += 1
        self._idx = self._idx % self.max_size

        self.size += 1
        self.size = min(self.size, self.max_size)

    def sample(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        idxs = np.random.choice(self.size, batch_size, replace=False)
        experiences = np.vstack(self.ss_mem[idxs]), \
                    np.vstack(self.as_mem[idxs]), \
                    np.vstack(self.rs_mem[idxs]), \
                    np.vstack(self.ps_mem[idxs]), \
                    np.vstack(self.ds_mem[idxs]) 
        return experiences

    def __len__(self):
        return self.size

class DQN:
    def __init__(self,
                replay_buffer_fn,
                value_model_fn,
                value_optimizer_fn,
                value_optimizer_lr: float,
                training_strategy_fn,
                evaluation_strategy_fn,
                n_warmup_batches: int,
                update_target_every_steps: int):
        self.replay_buffer_fn = replay_buffer_fn
        self.value_model_fn = value_model_fn
        self.value_optimizer_fn = value_optimizer_fn
        self.value_optimizer_lr = value_optimizer_lr
        self.training_strategy_fn = training_strategy_fn
        self.evaluation_strategy_fn = evaluation_strategy_fn
        self.n_warmup_batches = n_warmup_batches
        self.update_target_every_steps = update_target_every_steps

    def optimize_model(self, experiences) -> None:
        states, actions, rewards, next_states, is_terminals = experiences
    
        max_a_q_sp = self.target_model(next_states).detach().max(1)[0].unsqueeze(1)
        target_q_sa = rewards + (self.gamma * max_a_q_sp * (1 - is_terminals))
    
        #create loss closure for smb algorithm
        def closure():
            self.value_optimizer.zero_grad()
            q_sa = self.online_model(states).gather(1, actions)
            td_error = q_sa - target_q_sa
            value_loss = td_error.pow(2).mul(0.5).mean()
            return value_loss
            #value_loss.backward() #NOTE: This seems to be part of the optimizer step for SMB
            # forward pass
        self.value_optimizer.step(closure=closure)

    def interaction_step(self, state: np.ndarray, env: gym.Env):
        action = self.training_strategy.select_action(self.online_model, state)
        new_state, reward, is_terminal, info = env.step(action)
        is_truncated = 'TimeLimit.truncated' in info and info['TimeLimit.truncated']
        is_failure = is_terminal and not is_truncated
        experience = (state, action, reward, new_state, float(is_failure))

        self.replay_buffer.store(experience)
        self.episode_reward[-1] += reward
        self.episode_timestep[-1] += 1
        self.episode_exploration[-1] += \
                int(self.training_strategy.exploratory_action_taken)
        return new_state, is_terminal
    
    def update_network(self) -> None:
        for target, online in zip(self.target_model.parameters(),
                                self.online_model.parameters()):
            target.data.copy_(online.data)
    
    def train(self, make_env_fn, make_env_kargs,
                seed: int, gamma: float, max_minutes: float,
                max_episodes: int, goal_mean_100_reward: float):
        training_start = time.time()
        last_debug_time = float('-inf')
        self.make_env_fn = make_env_fn
        self.make_env_kargs = make_env_kargs
        self.seed = seed
        self.gamma = gamma

        env = self.make_env_fn(**self.make_env_kargs,
                        seed = self.seed)
        
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        nS = env.observation_space.shape[0]
        nA = env.action_space.n

        self.episode_timestep= []
        self.episode_reward = []
        self.episode_seconds = []
        self.evaluation_scores = []
        self.episode_exploration = []

        self.target_model = self.value_model_fn(nS, nA)
        self.online_model = self.value_model_fn(nS, nA)
        self.update_network()

        self.value_optimizer = \
            self.value_optimizer_fn(self.online_model, 
                        self.value_optimizer_lr)

        self.replay_buffer = self.replay_buffer_fn()
        self.training_strategy = self.training_strategy_fn()
        self.evaluation_strategy = self.evaluation_strategy_fn()

        result = np.empty((max_episodes, 5))
        result[:] = np.nan
        training_time = 0
        for episode in range(1, max_episodes + 1):
            episode_start = time.time()

            state = env.reset()
            is_terminal = False
            self.episode_reward.append(0.0)
            self.episode_timestep.append(0.0)
            self.episode_exploration.append(0.0)

            for step in count():
                state, is_terminal = self.interaction_step(state, env)

                min_samples = self.replay_buffer.batch_size * \
                                self.n_warmup_batches
                
                if len(self.replay_buffer) > min_samples:
                    experiences = self.replay_buffer.sample()
                    experiences = self.online_model.load(experiences)
                    self.optimize_model(experiences)

                if np.sum(self.episode_timestep) % self.update_target_every_steps == 0:
                    self.update_network()
                
                if is_terminal:
                    gc.collect()
                    break

            episode_elapsed = time.time() - episode_start
            self.episode_seconds.append(episode_elapsed)
            training_time += episode_elapsed
            evaluation_score, _ = self.evaluate(self.online_model, env)

            total_step = int(np.sum(self.episode_timestep))
            self.evaluation_scores.append(evaluation_score)

            mean_10_reward = np.mean(self.episode_reward[-10:])
            std_10_reward = np.std(self.episode_reward[-10:])
            mean_100_reward = np.mean(self.episode_reward[-100:])
            std_100_reward = np.std(self.episode_reward[-100:])
            mean_100_eval_score = np.mean(self.evaluation_scores[-100:])
            std_100_eval_score = np.std(self.evaluation_scores[-100:])
            lst_100_exp_rat = np.array(
                self.episode_exploration[-100:])/np.array(self.episode_timestep[-100:])
            mean_100_exp_rat = np.mean(lst_100_exp_rat)
            std_100_exp_rat = np.std(lst_100_exp_rat)

            wallclock_elapsed = time.time() - training_start
            result[episode-1] = total_step, mean_100_reward, \
                mean_100_eval_score, training_time, wallclock_elapsed
            
            reached_debug_time = time.time() - last_debug_time >= 60
            reached_max_minutes = wallclock_elapsed >= max_minutes * 60
            reached_max_episodes = episode >= max_episodes
            reached_goal_mean_reward = mean_100_eval_score >= goal_mean_100_reward
            training_is_over = reached_max_minutes or \
                               reached_max_episodes or \
                               reached_goal_mean_reward

            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - training_start))
            debug_message = 'el {}, ep {:04}, ts {:06}, '
            debug_message += 'ar 10 {:05.1f}\u00B1{:05.1f}, '
            debug_message += '100 {:05.1f}\u00B1{:05.1f}, '
            debug_message += 'ex 100 {:02.1f}\u00B1{:02.1f}, '
            debug_message += 'ev {:05.1f}\u00B1{:05.1f}'
            debug_message = debug_message.format(
                elapsed_str, episode-1, total_step, mean_10_reward, std_10_reward, 
                mean_100_reward, std_100_reward, mean_100_exp_rat, std_100_exp_rat,
                mean_100_eval_score, std_100_eval_score)
            print(debug_message, end='\r', flush=True)
            if reached_debug_time or training_is_over:
                print('\x1b[2K' + debug_message, flush=True)
                last_debug_time = time.time()
            if training_is_over:
                if reached_max_minutes: print(u'--> reached_max_minutes \u2715')
                if reached_max_episodes: print(u'--> reached_max_episodes \u2715')
                if reached_goal_mean_reward: print(u'--> reached_goal_mean_reward \u2713')
                break
        
        final_eval_score, score_std = self.evaluate(self.online_model, env, n_episodes=100)
        wallclock_time = time.time() - training_start
        print('Training complete')
        print('Final evaluation score {:.2f}\u00B1{:.2f} in {:.2f}s training time,'
              ' {:.2f}s wall-clock time.\n'.format(
                  final_eval_score, score_std, training_time, wallclock_time))
        env.close()
        del env
        return result, final_eval_score, training_time, wallclock_time

    def evaluate(self, eval_policy_model: nn.Module, eval_env: gym.Env, n_episodes: int = 1):
        rs= []
        for _ in range(n_episodes):
            s = eval_env.reset()
            d = False
            rs.append(0)
            for _ in count():
                a = self.evaluation_strategy.select_action(eval_policy_model, s)
                s, r, d, _ = eval_env.step(a)
                rs[-1] += r
                if d:
                    break
        return np.mean(rs), np.std(rs)
    
best_agent, best_eval_score = None, float('-inf')
seed = 42
environment_settings = {
    'env_name': 'CartPole-v1',
    'gamma': 1.00,
    'max_minutes': 10,
    'max_episodes': 10000,
    'goal_mean_100_reward': 475
}

value_model_fn = lambda nS, nA: FCQ(nS, nA, hidden_dims=(512, 128))
#value_optimizer_fn = lambda net, lr: optim.RMSprop(net.parameters(), lr=lr)
value_optimizer_fn = lambda net, lr: smb.SMB(net.parameters(), lr=lr, independent_batch=False) #NOTE: Added SMB optimizer here

value_optimizer_lr = 0.5 #NOTE: Adjusted to 0.5 (default in paper)

training_strategy_fn = lambda: EGreedyStrategy(epsilon=0.1)
evaluation_strategy_fn = lambda: GreedyStrategy()

replay_buffer_fn = lambda: ReplayBuffer(max_size=50000, batch_size=64)
n_warmup_batches = 5
update_target_every_steps = 10

env_name, gamma, max_minutes, max_episodes, goal_mean_100_reward = environment_settings.values()
agent = DQN(replay_buffer_fn, 
            value_model_fn,
            value_optimizer_fn,
            value_optimizer_lr,
            training_strategy_fn,
            evaluation_strategy_fn,
            n_warmup_batches,
            update_target_every_steps)

make_env_fn, make_env_kargs = get_make_env_fn(env_name=env_name)
result, final_eval_score, training_time, wallclock_time = agent.train(
            make_env_fn, make_env_kargs, seed, gamma, max_minutes, max_episodes, goal_mean_100_reward)

print('Done')