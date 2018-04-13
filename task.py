from Log import Log
import numpy as np
from physics_sim_fixed import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=10., target_pos=None, log=None):
        """Initialize a Task object.
        Params
        ======
            log (Log): Reference to the log utility.
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode - originally only 5
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 1
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10., 0., 0., 0.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        alpha = 0.2
        max_reward = 1.0
        pos_reward = 10.
        reward_spread = 1.0
        #reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()  # Original
        #reward = 1 / np.exp(np.sum(np.abs(self.sim.pose - self.target_pos)*alpha))  #v1
        #reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()  #v2
        #reward = 1.-alpha*(abs(self.sim.pose - self.target_pos)).sum()  #v3
        #reward = 30.-alpha*( pos_reward*(abs(self.sim.pose[:3] - self.target_pos[:3])) + abs(self.sim.pose[3:] - self.target_pos[3:])).sum()  #v4, 5
        #reward = 1 / np.exp(np.sum(np.abs(self.sim.pose - self.target_pos)*alpha))  #v6
        #reward = np.tanh(max_reward - np.sqrt(np.abs(self.sim.pose - self.target_pos).sum()))  #v7
        #reward = 1.-alpha*((self.target_pos - self.sim.pose) / reward_spread).sum()  #v8
        #reward = np.tanh(1.-alpha*((self.target_pos - self.sim.pose) / reward_spread).sum())  #v9
        #reward = np.tanh(max_reward-alpha*(abs(self.sim.pose - self.target_pos)).sum()) / reward_spread  #v10
        #reward = np.tanh((max_reward-alpha*(abs(self.sim.pose - self.target_pos)).sum()) / reward_spread)  #v10
        #reward = np.tanh((max_reward-alpha*(pos_reward * abs(self.sim.pose[:3] - self.target_pos[:3]) + abs(self.sim.pose[3:] - self.target_pos[3:])).sum()) / reward_spread)  #v11
        reward = np.tanh(max_reward-alpha*((abs(self.sim.pose - self.target_pos)).sum())) / reward_spread  #v12
        #reward += pos_reward - np.abs(self.sim.pose[2] - self.target_pos[2])  #Special Z axis reward
        #reward = np.tanh((max_reward-alpha*((abs(self.sim.pose - self.target_pos)).sum())) / reward_spread)  #v13
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state