import numpy as np
from physics_sim import PhysicsSim
import math

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        # state is based on pose, velocities and angle_velocities
        self.state_size = self.action_repeat * 6
        
        # simplify the Action space to 4 delta rotor speeds [0,1,2,3] and the mean rotor speeds [4].
        self.action_low = np.array([0., 0., 0., 0., 0.])
        self.action_high = np.array([10., 10., 10., 10., 895])
        self.action_size = 5

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 
        
        self.maxRange = [self.sim.upper_bounds[:3] - self.target_pos[:3], self.target_pos[:3] - self.sim.lower_bounds[:3]]
        self.maxRange = np.max(self.maxRange,axis=0)
        print(self.maxRange)

    def get_reward(self, rotor_speeds):
        """Uses current pose of sim to return reward."""

        s = (self.target_pos - self.sim.pose[:3])
        v = self.sim.v
        a = self.sim.linear_accel
        sv = np.dot(s, v) 
        sa = np.dot(s, a) 


        d_normalized = np.linalg.norm(np.linalg.norm(s/self.maxRange))
        d_reward = 1. - 2.*np.power(d_normalized, .5)
        
        
        # The quadcopter has a tendency to get stuck in the local minimum pos [0, 0, 0]
        # when simplify the Action space to 5 actions: 4 delta rotor speeds plus mean rotor speeds.
        # The purpose of local_minimum_discount is to push the quadcopter away from this point
        local_minimum = self.sim.pose[:3] - np.array([self.target_pos[0], self.target_pos[1], 0.])
        local_minimum = np.linalg.norm(local_minimum / [2., 2., 8.])
        local_minimum_discount = np.power(1. - np.clip(1. - local_minimum, 0, 1), 4.)
        
        
        reward = d_reward * local_minimum_discount
        
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward(rotor_speeds) 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        pose_all = []
        pose_all.append(self.sim.pose)
        state = np.concatenate(pose_all * self.action_repeat) 
        return state