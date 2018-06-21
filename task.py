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
        #self.state_size = self.action_repeat * (6 + 6)
        self.state_size = self.action_repeat * (6 + 3)
        #self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self, rotor_speeds):
        """Uses current pose of sim to return reward."""

        s = (self.target_pos - self.sim.pose[:3])
        v = self.sim.v
        a = self.sim.linear_accel
        sv = np.dot(s, v) 
        sa = np.dot(s, a) 

        revard_s_close = np.clip(1. - np.linalg.norm(s*[.1, .1, .05]), 0, 1)
        revard_s = np.clip(1. - np.linalg.norm(s)*.01, 0, 1)
        revard_sv = np.clip(sv*0.01, 0, 1)
        reward = 1. + revard_s_close + revard_s + revard_sv
            
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward(rotor_speeds) 
            pose_all.append(self.sim.pose)
            pose_all.append(self.sim.v)
            #pose_all.append(self.sim.angular_v)
        next_state = np.concatenate(pose_all)
        
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        pose_all = []
        pose_all.append(self.sim.pose)
        pose_all.append(self.sim.v)
        #pose_all.append(self.sim.angular_v)
        state = np.concatenate(pose_all * self.action_repeat) 
        return state