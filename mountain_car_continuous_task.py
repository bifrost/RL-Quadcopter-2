import gym
import numpy as np

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self):
        """Initialize a Task object.
        """
        # Simulation
        self.sim = gym.make('MountainCarContinuous-v0')

        self.state_size = self.sim.observation_space.shape[0]
        self.action_size = self.sim.action_space.shape[0]
        self.action_low = self.sim.action_space.low
        self.action_high = self.sim.action_space.high
        
        print('state_size', self.state_size)
        print('action_size', self.action_size)
        print('action_low', self.action_low)
        print('action_high', self.action_high)

    def step(self, action):
        """Uses action to obtain next state, reward, done."""
        state, reward, done, _  = self.sim.step(action)
        return state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        return self.sim.reset()