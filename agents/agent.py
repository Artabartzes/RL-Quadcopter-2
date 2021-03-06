from agents.Actor import Actor
from agents.Critic import Critic
from OUNoise import OUNoise
from ReplayBuffer import ReplayBuffer
import numpy as np
import csv

class DDPG():
    """Reinforcement Learning agent that learns using DDPG.

        Params
        ======
            task (Task): Instance of the Task class which reports the environment to this agent
            log (Log): Reference to the log utility.
    """
    def __init__(self, task, log=None):
        self.task = task
        #Add log utility
        self.log = log
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high

        # Actor (Policy) Model
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high, self.log)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high, self.log)

        # Critic (Value) Model
        self.critic_local = Critic(self.state_size, self.action_size, self.log)
        self.critic_target = Critic(self.state_size, self.action_size, self.log)

        # Initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        # Noise process
        # @hakimka on the forums said to use mu = 0.001, theta = 0.05, sigma = 0.0015, but...
        self.exploration_mu = 0.001
        self.exploration_theta = 0.05
        self.exploration_sigma = 0.0015
        self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)

        # Replay memory
        self.buffer_size = 100000
        self.batch_size = 128
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size, self.log)

        # Algorithm parameters
        self.gamma = 0.99  # discount factor 0.99
        self.tau = 0.01  # for soft update of target parameters  0.01

        # score
        self.total_reward = 0.0
        self.count = 0
        self.score = 0
        self.best_score = -np.inf

    def reset_episode(self):
        self.noise.reset()
        state = self.task.reset()
        self.last_state = state
        self.total_reward = 0.0
        self.count = 0
        return state

    def step(self, action, reward, next_state, done):
         # Save experience / reward
        self.memory.add(self.last_state, action, reward, next_state, done)
        self.total_reward += reward
        self.count += 1
        #if self.log != None:
        #    self.log.write('DDPG.step len(self.memory)=' + str(len(self.memory)) + \
        #        ' total_reward=' + str(self.total_reward) + ' count=' + str(self.count))

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            #if self.log != None:
            #    self.log.write('DDPG.step len(self.memory) > self.batch_size' + \
            #        str(len(self.memory)) + '>' + str(self.batch_size))
            self.learn(experiences)
        #print(self.critic_local.model.get_weights()[0][0])
        #print(self.critic_target.model.get_weights()[0][0])
        #for lay in self.critic_local.model.layers:
        #    if lay.name == 'q_values':
        #        print(lay.name + ': ' + str(lay.get_weights()[0][5]))

        # Roll over last state and action
        self.last_state = next_state

    def act(self, state):
        """Returns actions for given state(s) as per current policy."""
        state = np.reshape(state, [-1, self.state_size])
        action = self.actor_local.model.predict(state)[0]
        return list(action + self.noise.sample())  # add some noise for exploration

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples."""
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        #if self.log != None:
        #    self.log.write('DDPG.learn experiences=' + str(len(experiences)))
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # Get predicted next-state actions and Q values from target models
        #     Q_targets_next = critic_target(next_state, actor_target(next_state))
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        #actions_next_normal = ( actions_next - self.action_low ) / self.action_high
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])
        #Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next_normal])

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        #actions_normal = ( actions - self.action_low ) / self.action_high
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)
        #self.critic_local.model.train_on_batch(x=[states, actions_normal], y=Q_targets)

        # Train actor model (local)
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        #action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions_normal, 0]), (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])  # custom training function
        if self.log != None:
            self.log.write('DDPG.learn Q_targets=' + str(Q_targets))
            self.log.write('DDPG.learn action_gradients=' + str(action_gradients))

        # Soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)

        self.score = self.total_reward / float(self.count) if self.count else 0.0
        self.best_score = max(self.best_score, self.score)
        #if self.score > self.best_score:
        #    self.best_score = self.score

    def soft_update(self, local_model, target_model):
        """Soft update model parameters. As opposed to a fixed Q-targets method."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size"

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)