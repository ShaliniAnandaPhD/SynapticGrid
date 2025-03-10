"""
Traffic Signal AI
================
Adaptive traffic light controller using reinforcement learning to minimize congestion.
This module simulates, trains, and deploys AI controllers for traffic intersections.
"""

import numpy as np
import pandas as pd
import gym
from gym import spaces
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import random
import time
import os
import json
import logging
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("traffic_ai.log"), logging.StreamHandler()]
)
logger = logging.getLogger("TrafficSignalAI")

# Traffic light phases (simplification of actual traffic light patterns)
PHASES = {
    0: "North-South Green, East-West Red",
    1: "North-South Yellow, East-West Red",
    2: "North-South Red, East-West Green",
    3: "North-South Red, East-West Yellow"
}

# Vehicle types with their properties
VEHICLE_TYPES = {
    'car': {'length': 4.5, 'max_speed': 15.0, 'accel': 2.5, 'decel': 4.5},
    'bus': {'length': 12.0, 'max_speed': 12.0, 'accel': 1.5, 'decel': 3.0},
    'truck': {'length': 8.0, 'max_speed': 13.0, 'accel': 1.8, 'decel': 3.5},
    'motorcycle': {'length': 2.0, 'max_speed': 18.0, 'accel': 3.5, 'decel': 5.0},
    'bicycle': {'length': 1.8, 'max_speed': 6.0, 'accel': 1.2, 'decel': 2.0}
}


class Vehicle:
    """
    Represents a vehicle in the traffic simulation.
    
    Each vehicle has position, speed, and other properties that evolve
    as it moves through the traffic network.
    """
    
    next_id = 0  # Class variable for unique vehicle IDs
    
    def __init__(self, lane: str, position: float = 0.0, speed: float = 0.0, 
                vehicle_type: str = 'car', destination: str = None):
        """
        Initialize a vehicle.
        
        Args:
            lane: Current lane ID (e.g., 'north_in_1')
            position: Position along the lane in meters (0 = start of lane)
            speed: Current speed in m/s
            vehicle_type: Type of vehicle ('car', 'bus', etc.)
            destination: Target lane (for routing)
        """
        # Get unique ID
        self.id = Vehicle.next_id
        Vehicle.next_id += 1
        
        # Location and movement
        self.lane = lane
        self.position = position
        self.speed = speed
        self.acceleration = 0.0
        
        # Route and status
        self.destination = destination
        self.route = []  # List of lanes to follow
        self.waiting_time = 0.0  # Time spent waiting at lights
        self.total_travel_time = 0.0  # Total time in the network
        self.arrived = False
        
        # Verify and set vehicle type
        if vehicle_type in VEHICLE_TYPES:
            self.type = vehicle_type
            self.properties = VEHICLE_TYPES[vehicle_type].copy()
        else:
            self.type = 'car'
            self.properties = VEHICLE_TYPES['car'].copy()
            logger.warning(f"Unknown vehicle type {vehicle_type}, defaulting to car")
    
    def update(self, dt: float, traffic_light_state: int, leader_distance: float = float('inf'),
              leader_speed: float = None):
        """
        Update vehicle state for one timestep.
        
        Args:
            dt: Time step in seconds
            traffic_light_state: Current state of the next traffic light
            leader_distance: Distance to the vehicle ahead (if any)
            leader_speed: Speed of the vehicle ahead (if any)
            
        Returns:
            Updated vehicle state
        """
        # Check if already arrived
        if self.arrived:
            return
            
        # Update travel time
        self.total_travel_time += dt
        
        # Calculate new acceleration, speed, and position
        self._update_acceleration(traffic_light_state, leader_distance, leader_speed)
        
        # Apply acceleration to update speed (with limits)
        self.speed += self.acceleration * dt
        self.speed = max(0.0, min(self.speed, self.properties['max_speed']))
        
        # Update position
        old_position = self.position
        self.position += self.speed * dt
        
        # Check if waiting (very slow speed)
        if self.speed < 0.5:  # Less than 0.5 m/s is considered waiting
            self.waiting_time += dt
            
    def _update_acceleration(self, traffic_light_state: int, leader_distance: float,
                           leader_speed: float = None):
        """
        Calculate vehicle acceleration based on traffic conditions.
        
        This uses a simplified Intelligent Driver Model (IDM) for car-following
        and traffic light responses.
        
        Args:
            traffic_light_state: Current state of the next traffic light
            leader_distance: Distance to the vehicle ahead
            leader_speed: Speed of the vehicle ahead
            
        Returns:
            Updated acceleration value
        """
        # Maximum acceleration in free traffic
        free_road_accel = self.properties['accel']
        
        # Desired speed - try to reach max speed when possible
        v_desired = self.properties['max_speed']
        acceleration = free_road_accel * (1 - (self.speed / v_desired)**4)
        
        # Adjust for leader vehicle (if any) - simple car-following model
        if leader_distance < float('inf'):
            # Safety distance = min gap + time headway * speed
            min_gap = self.properties['length'] + 1.0  # 1m minimum gap
            time_headway = 1.5  # 1.5 seconds time gap
            desired_gap = min_gap + time_headway * self.speed
            
            # If we're too close to the leader
            if leader_distance < desired_gap:
                # If leader info available, use it
                if leader_speed is not None:
                    # Deceleration to maintain safe distance
                    decel = ((self.speed - leader_speed)**2) / (2 * leader_distance)
                    acceleration -= min(decel, self.properties['decel'])
                else:
                    # Conservative deceleration if we don't know leader speed
                    acceleration = -self.properties['decel'] * (desired_gap / leader_distance)**2
        
        # Adjust for traffic lights
        # Check if yellow or red for our direction (simplified)
        light_distance = 50.0  # for example, distance to next traffic light
        
        # Simple check if the light is red/yellow for our direction
        # This is a simplification - in reality would depend on lane and exact state
        is_red_or_yellow = ((self.lane.startswith('north') or self.lane.startswith('south')) and 
                           traffic_light_state >= 1) or \
                          ((self.lane.startswith('east') or self.lane.startswith('west')) and 
                           (traffic_light_state == 0 or traffic_light_state == 1))
        
        if is_red_or_yellow and light_distance < 50:
            # Time to reach light at current speed
            time_to_light = light_distance / max(0.1, self.speed)
            
            # If we can't make it through before red, slow down
            if time_to_light > 2.0:  # if more than 2 seconds to light
                stopping_decel = (self.speed**2) / (2 * light_distance)
                acceleration = min(acceleration, -stopping_decel)
        
        # Apply limits
        self.acceleration = max(-self.properties['decel'], 
                              min(self.properties['accel'], acceleration))
    
    def __repr__(self) -> str:
        return f"Vehicle(id={self.id}, type={self.type}, lane={self.lane}, pos={self.position:.1f}m, speed={self.speed:.1f}m/s)"


class Lane:
    """
    Represents a single lane of traffic.
    
    Lanes connect intersections and contain vehicles.
    """
    
    def __init__(self, lane_id: str, length: float, max_speed: float,
                origin: str = None, destination: str = None):
        """
        Initialize a lane.
        
        Args:
            lane_id: Unique lane identifier
            length: Length of lane in meters
            max_speed: Maximum speed limit in m/s
            origin: Start intersection ID
            destination: End intersection ID
        """
        self.id = lane_id
        self.length = length
        self.max_speed = max_speed
        self.origin = origin
        self.destination = destination
        
        # List of vehicles in the lane (sorted by position)
        self.vehicles = []
        
        # Traffic counts
        self.vehicles_entered = 0
        self.vehicles_exited = 0
        
        # Occupancy tracking (for congestion measurement)
        self.occupancy_history = []
    
    def add_vehicle(self, vehicle: Vehicle) -> None:
        """Add a vehicle to this lane and set its initial properties."""
        # Set the vehicle's lane and initial position
        vehicle.lane = self.id
        
        # If lane is 'incoming', place at start, otherwise place at end (for testing)
        if 'in' in self.id:
            vehicle.position = 0.0
        else:
            vehicle.position = self.length
            
        # Add to lane and sort vehicles by position
        self.vehicles.append(vehicle)
        self.vehicles.sort(key=lambda v: v.position)
        
        # Update counts
        self.vehicles_entered += 1
    
    def remove_vehicle(self, vehicle: Vehicle) -> None:
        """Remove a vehicle from this lane."""
        if vehicle in self.vehicles:
            self.vehicles.remove(vehicle)
            self.vehicles_exited += 1
    
    def update(self, dt: float, traffic_light_state: int) -> List[Vehicle]:
        """
        Update all vehicles in the lane for one timestep.
        
        Args:
            dt: Time step in seconds
            traffic_light_state: Current state of the traffic light
            
        Returns:
            List of vehicles that have exited the lane
        """
        exited_vehicles = []
        
        # Update vehicles from back to front so we know leader positions
        for i in range(len(self.vehicles) - 1, -1, -1):
            vehicle = self.vehicles[i]
            
            # Determine leader info
            leader_distance = float('inf')
            leader_speed = None
            
            if i < len(self.vehicles) - 1:  # If not the first vehicle
                leader = self.vehicles[i + 1]
                leader_distance = leader.position - vehicle.position - vehicle.properties['length']
                leader_speed = leader.speed
            
            # Update vehicle
            vehicle.update(dt, traffic_light_state, leader_distance, leader_speed)
            
            # Check if vehicle has exited the lane
            if vehicle.position >= self.length:
                exited_vehicles.append(vehicle)
        
        # Remove exited vehicles
        for vehicle in exited_vehicles:
            self.remove_vehicle(vehicle)
        
        # Calculate occupancy (percent of lane physically occupied by vehicles)
        total_vehicle_length = sum(v.properties['length'] for v in self.vehicles)
        occupancy = min(1.0, total_vehicle_length / self.length)
        self.occupancy_history.append(occupancy)
        
        # Keep history limited to last 100 steps
        if len(self.occupancy_history) > 100:
            self.occupancy_history = self.occupancy_history[-100:]
            
        return exited_vehicles
    
    def get_average_occupancy(self, window: int = 10) -> float:
        """Calculate average occupancy over the last n timesteps."""
        if not self.occupancy_history:
            return 0.0
        
        window = min(window, len(self.occupancy_history))
        return sum(self.occupancy_history[-window:]) / window
    
    def get_average_speed(self) -> float:
        """Calculate average speed of all vehicles in the lane."""
        if not self.vehicles:
            return self.max_speed  # Empty lane = free flow
            
        return sum(v.speed for v in self.vehicles) / len(self.vehicles)
    
    def __repr__(self) -> str:
        return f"Lane(id={self.id}, length={self.length}m, vehicles={len(self.vehicles)})"


class Intersection:
    """
    Represents a traffic intersection with connected lanes and traffic signals.
    """
    
    def __init__(self, intersection_id: str, incoming_lanes: List[str] = None,
                outgoing_lanes: List[str] = None):
        """
        Initialize an intersection.
        
        Args:
            intersection_id: Unique identifier
            incoming_lanes: List of incoming lane IDs
            outgoing_lanes: List of outgoing lane IDs
        """
        self.id = intersection_id
        self.incoming_lanes = incoming_lanes or []
        self.outgoing_lanes = outgoing_lanes or []
        
        # Traffic signal state and timing
        self.phase = 0  # Current phase
        self.phase_time = 0.0  # Time in current phase
        self.phase_durations = [30.0, 3.0, 30.0, 3.0]  # Default durations for each phase
        
        # Stats for monitoring
        self.total_wait_time = 0.0
        self.total_vehicles = 0
        self.throughput_history = []
    
    def update(self, dt: float, new_phase: int = None) -> None:
        """
        Update the intersection state for one timestep.
        
        Args:
            dt: Time step in seconds
            new_phase: Optional new phase to set
        """
        # If a new phase is requested, set it
        if new_phase is not None and new_phase != self.phase:
            self.phase = new_phase
            self.phase_time = 0.0
        else:
            # Otherwise, update phase time and check for phase transition
            self.phase_time += dt
            
            # Check if time to change to next phase
            if self.phase_time >= self.phase_durations[self.phase]:
                self.phase = (self.phase + 1) % len(self.phase_durations)
                self.phase_time = 0.0
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the intersection.
        
        Returns:
            Dict with phase, phase_time, and other relevant info
        """
        return {
            'phase': self.phase,
            'phase_time': self.phase_time,
            'phase_duration': self.phase_durations[self.phase],
            'phase_description': PHASES[self.phase]
        }
    
    def set_phase_durations(self, durations: List[float]) -> None:
        """Set new phase durations."""
        if len(durations) != len(self.phase_durations):
            logger.warning(f"Expected {len(self.phase_durations)} durations but got {len(durations)}")
            return
            
        self.phase_durations = durations
    
    def __repr__(self) -> str:
        return f"Intersection(id={self.id}, phase={self.phase}, phase_time={self.phase_time:.1f}s)"


class TrafficEnvironment:
    """
    Traffic environment for the reinforcement learning agent.
    
    Simulates a network of intersections, lanes, and vehicles.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the traffic environment.
        
        Args:
            config: Dictionary with configuration parameters
        """
        # Set default config if none provided
        self.config = config or {
            'duration': 3600,  # Simulation duration in seconds
            'timestep': 1.0,   # Simulation timestep in seconds
            'max_vehicles': 1000,  # Maximum number of vehicles
            'spawn_rate': 0.1,  # Vehicle spawn probability per second
            'seed': 42,  # Random seed
        }
        
        # Set random seed
        random.seed(self.config['seed'])
        np.random.seed(self.config['seed'])
        
        # Initialize intersections and lanes
        self.intersections = {}
        self.lanes = {}
        
        # Initialize time
        self.time = 0.0
        self.step_count = 0
        
        # Stats and history
        self.stats = {
            'total_wait_time': 0.0,
            'total_travel_time': 0.0,
            'total_vehicles': 0,
            'completed_trips': 0,
            'throughput': 0,
        }
        self.history = {
            'wait_times': [],
            'throughput': [],
            'queue_lengths': []
        }
        
        # Setup default network if no config provided
        if 'network' not in self.config:
            self._setup_default_network()
        else:
            self._load_network(self.config['network'])
            
        logger.info(f"Traffic environment initialized with {len(self.intersections)} "
                  f"intersections and {len(self.lanes)} lanes")
    
    def _setup_default_network(self) -> None:
        """
        Create a simple default traffic network.
        
        This creates a basic 4-way intersection with incoming and outgoing lanes.
        """
        # Create a single intersection
        intersection = Intersection('central', 
                                   incoming_lanes=['north_in', 'south_in', 'east_in', 'west_in'],
                                   outgoing_lanes=['north_out', 'south_out', 'east_out', 'west_out'])
        self.intersections['central'] = intersection
        
        # Create lanes
        for direction in ['north', 'south', 'east', 'west']:
            # Incoming lane (towards intersection)
            self.lanes[f'{direction}_in'] = Lane(
                lane_id=f'{direction}_in',
                length=500.0,  # 500 meters
                max_speed=13.9,  # 50 km/h in m/s
                destination='central'
            )
            
            # Outgoing lane (away from intersection)
            self.lanes[f'{direction}_out'] = Lane(
                lane_id=f'{direction}_out',
                length=500.0,
                max_speed=13.9,
                origin='central'
            )
            
        logger.info("Created default 4-way intersection network")
    
    def _load_network(self, network_config: Dict[str, Any]) -> None:
        """
        Load a network from a configuration dictionary.
        
        Args:
            network_config: Dict with intersections and lanes definitions
        """
        # Load intersections
        for intersection_id, data in network_config.get('intersections', {}).items():
            self.intersections[intersection_id] = Intersection(
                intersection_id=intersection_id,
                incoming_lanes=data.get('incoming_lanes', []),
                outgoing_lanes=data.get('outgoing_lanes', [])
            )
            
            # Set phase durations if provided
            if 'phase_durations' in data:
                self.intersections[intersection_id].set_phase_durations(data['phase_durations'])
        
        # Load lanes
        for lane_id, data in network_config.get('lanes', {}).items():
            self.lanes[lane_id] = Lane(
                lane_id=lane_id,
                length=data.get('length', 500.0),
                max_speed=data.get('max_speed', 13.9),
                origin=data.get('origin'),
                destination=data.get('destination')
            )
            
        logger.info(f"Loaded network with {len(self.intersections)} intersections "
                  f"and {len(self.lanes)} lanes")
    
    def reset(self) -> np.ndarray:
        """
        Reset the environment to initial state.
        
        Returns:
            Initial observation
        """
        # Reset time
        self.time = 0.0
        self.step_count = 0
        
        # Clear all lanes (remove vehicles)
        for lane in self.lanes.values():
            lane.vehicles = []
            lane.vehicles_entered = 0
            lane.vehicles_exited = 0
            lane.occupancy_history = []
        
        # Reset intersections
        for intersection in self.intersections.values():
            intersection.phase = 0
            intersection.phase_time = 0.0
            intersection.total_wait_time = 0.0
            intersection.total_vehicles = 0
            intersection.throughput_history = []
        
        # Reset stats
        self.stats = {
            'total_wait_time': 0.0,
            'total_travel_time': 0.0,
            'total_vehicles': 0,
            'completed_trips': 0,
            'throughput': 0,
        }
        self.history = {
            'wait_times': [],
            'throughput': [],
            'queue_lengths': []
        }
        
        # Spawn initial vehicles
        self._spawn_vehicles()
        
        # Get initial observation
        return self._get_observation()
    
    def step(self, actions: Dict[str, int] = None) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            actions: Dict mapping intersection IDs to phase actions
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        # Default empty actions if none provided
        if actions is None:
            actions = {}
            
        # Apply actions to intersections
        for intersection_id, action in actions.items():
            if intersection_id in self.intersections:
                self.intersections[intersection_id].update(
                    self.config['timestep'], new_phase=action
                )
            else:
                logger.warning(f"Action provided for unknown intersection {intersection_id}")
        
        # Update intersections without explicit actions
        for intersection_id, intersection in self.intersections.items():
            if intersection_id not in actions:
                intersection.update(self.config['timestep'])
        
        # Update all lanes
        throughput = 0
        for lane_id, lane in self.lanes.items():
            # Get the traffic light state affecting this lane
            traffic_light_state = 0  # Default
            
            # If lane has a destination intersection, get its state
            if lane.destination and lane.destination in self.intersections:
                traffic_light_state = self.intersections[lane.destination].phase
            
            # Update the lane and get exited vehicles
            exited_vehicles = lane.update(self.config['timestep'], traffic_light_state)
            throughput += len(exited_vehicles)
            
            # Handle exited vehicles (transfer to next lane or remove)
            for vehicle in exited_vehicles:
                # If the vehicle has a route, follow it
                if vehicle.route and len(vehicle.route) > 0:
                    next_lane_id = vehicle.route.pop(0)
                    if next_lane_id in self.lanes:
                        self.lanes[next_lane_id].add_vehicle(vehicle)
                    else:
                        logger.warning(f"Vehicle tried to enter unknown lane {next_lane_id}")
                        # Count as completed anyway
                        self.stats['completed_trips'] += 1
                else:
                    # No route or end of route = completed trip
                    self.stats['completed_trips'] += 1
                    self.stats['total_wait_time'] += vehicle.waiting_time
                    self.stats['total_travel_time'] += vehicle.total_travel_time
        
        # Spawn new vehicles
        new_vehicles = self._spawn_vehicles()
        self.stats['total_vehicles'] += new_vehicles
        
        # Update time
        self.time += self.config['timestep']
        self.step_count += 1
        
        # Record history
        self.stats['throughput'] = throughput
        self.history['throughput'].append(throughput)
        
        total_queue = sum(1 for lane in self.lanes.values() 
                        for vehicle in lane.vehicles if vehicle.speed < 0.5)
        self.history['queue_lengths'].append(total_queue)
        
        # Check if simulation is done
        done = self.time >= self.config['duration']
        
        # Get observation and reward
        observation = self._get_observation()
        reward = self._calculate_reward()
        
        # Compile info dictionary
        info = {
            'time': self.time,
            'stats': self.stats.copy(),
            'queues': {lane_id: len([v for v in lane.vehicles if v.speed < 0.5]) 
                     for lane_id, lane in self.lanes.items()},
            'throughput': throughput
        }
        
        return observation, reward, done, info
    
    def _spawn_vehicles(self) -> int:
        """
        Spawn new vehicles at entrances to the network.
        
        Returns:
            Number of vehicles spawned
        """
        num_spawned = 0
        
        # Find source lanes (no origin intersection)
        source_lanes = [lane_id for lane_id, lane in self.lanes.items() 
                      if lane.origin is None]
        
        # For each source lane, decide whether to spawn a vehicle
        for lane_id in source_lanes:
            # Check spawn probability
            if random.random() < self.config['spawn_rate'] * self.config['timestep']:
                # Check if lane has room (no vehicle near the start)
                lane = self.lanes[lane_id]
                
                # Only spawn if no vehicle in first 10m of lane
                vehicle_in_spawn_area = any(v.position < 10.0 for v in lane.vehicles)
                if not vehicle_in_spawn_area:
                    # Pick a random vehicle type with weights
                    vehicle_type = random.choices(
                        ['car', 'bus', 'truck', 'motorcycle', 'bicycle'],
                        weights=[0.75, 0.08, 0.12, 0.04, 0.01],
                        k=1
                    )[0]
                    
                    # Create vehicle
                    vehicle = Vehicle(
                        lane=lane_id,
                        position=0.0,
                        speed=lane.max_speed * 0.3,  # Start at 30% of max speed
                        vehicle_type=vehicle_type
                    )
                    
                    # Pick a random destination (any outgoing lane)
                    dest_lanes = [lane_id for lane_id, lane in self.lanes.items() 
                                if lane.destination is None]
                    if dest_lanes:
                        destination = random.choice(dest_lanes)
                        vehicle.destination = destination
                        
                        # Set a simple route (for now just source -> destination)
                        # In a real system, we'd use a routing algorithm here
                        vehicle.route = [destination]
                    
                    # Add to the lane
                    lane.add_vehicle(vehicle)
                    num_spawned += 1
        
        return num_spawned
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        Get the current observation of the traffic state.
        
        Returns:
            Dictionary mapping intersection IDs to their observations
        """
        observations = {}
        
        # For each intersection, create an observation
        for intersection_id, intersection in self.intersections.items():
            # Get queue lengths for incoming lanes
            queue_lengths = []
            for lane_id in intersection.incoming_lanes:
                if lane_id in self.lanes:
                    lane = self.lanes[lane_id]
                    # Count vehicles at or near stop line
                    queue = sum(1 for v in lane.vehicles if v.speed < 0.5)
                    queue_lengths.append(queue)
                else:
                    queue_lengths.append(0)
            
            # Get current phase and phase time
            phase = intersection.phase
            phase_time = intersection.phase_time
            phase_remaining = intersection.phase_durations[phase] - phase_time
            
            # Normalize phase time to [0, 1]
            norm_phase_time = phase_time / intersection.phase_durations[phase]
            
            # Get incoming lane average speeds
            avg_speeds = []
            for lane_id in intersection.incoming_lanes:
                if lane_id in self.lanes:
                    avg_speed = self.lanes[lane_id].get_average_speed()
                    # Normalize by max speed
                    norm_speed = avg_speed / self.lanes[lane_id].max_speed
                    avg_speeds.append(norm_speed)
                else:
                    avg_speeds.append(1.0)  # Default to free flow
            
            # Combine into observation vector
            obs = np.array(
                queue_lengths + [phase, norm_phase_time] + avg_speeds,
                dtype=np.float32
            )
            
            observations[intersection_id] = obs
            
        return observations
    
    def _calculate_reward(self) -> float:
        """
        Calculate the reward based on traffic state.
        
        The reward is negative and based on queue lengths, wait times,
        and throughput.
        
        Returns:
            Reward value (negative = cost)
        """
        # Calculate total queue length
        total_queue = sum(1 for lane in self.lanes.values() 
                        for vehicle in lane.vehicles if vehicle.speed < 0.5)
        
        # Calculate average travel time of completed trips
        avg_travel_time = 0.0
        if self.stats['completed_trips'] > 0:
            avg_travel_time = self.stats['total_travel_time'] / self.stats['completed_trips']
        
        # Calculate reward components
        queue_penalty = -0.1 * total_queue
        wait_penalty = -0.01 * avg_travel_time
        throughput_reward = 1.0 * self.stats['throughput']
        
        # Combine into final reward
        reward = queue_penalty + wait_penalty + throughput_reward
        
        return reward
    
    def render(self, mode: str = 'human') -> None:
        """
        Render the environment.
        
        Args:
            mode: Rendering mode
        """
        if mode == 'human':
            # Print simple text representation
            print(f"\nTime: {self.time:.1f}s")
            
            for intersection_id, intersection in self.intersections.items():
                phase_desc = PHASES[intersection.phase]
                print(f"Intersection {intersection_id}: {phase_desc}, "
                     f"Time in phase: {intersection.phase_time:.1f}s")
            
            for lane_id, lane in self.lanes.items():
                queue = sum(1 for v in lane.vehicles if v.speed < 0.5)
                print(f"Lane {lane_id}: {len(lane.vehicles)} vehicles, {queue} in queue")
                
            print(f"Throughput: {self.stats['throughput']} vehicles")
            print(f"Completed trips: {self.stats['completed_trips']}")
    
    def get_traffic_metrics(self) -> Dict[str, Any]:
        """
        Calculate comprehensive traffic metrics.
        
        Returns:
            Dictionary of traffic metrics
        """
        metrics = {
            'time': self.time,
            'completed_trips': self.stats['completed_trips'],
            'total_vehicles': self.stats['total_vehicles'],
            'completion_rate': self.stats['completed_trips'] / max(1, self.stats['total_vehicles']),
            'avg_travel_time': self.stats['total_travel_time'] / max(1, self.stats['completed_trips']),
            'avg_wait_time': self.stats['total_wait_time'] / max(1, self.stats['completed_trips']),
            'throughput': self.stats['throughput'],
            'avg_queues': np.mean(self.history['queue_lengths']),
            'max_queue': max(self.history['queue_lengths']),
            'lane_metrics': {}
        }
        
        # Calculate metrics for each lane
        for lane_id, lane in self.lanes.items():
            metrics['lane_metrics'][lane_id] = {
                'avg_occupancy': lane.get_average_occupancy(),
                'avg_speed': lane.get_average_speed(),
                'flow': lane.vehicles_exited / max(1, self.time / 3600)  # vehicles per hour
            }
            
        return metrics


class TrafficSignalGymEnv(gym.Env):
    """
    OpenAI Gym environment wrapper for traffic signal control.
    
    This provides a standardized interface for reinforcement learning algorithms.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the gym environment.
        
        Args:
            config: Dictionary with configuration parameters
        """
        super(TrafficSignalGymEnv, self).__init__()
        
        # Create traffic environment
        self.env = TrafficEnvironment(config)
        
        # Define action space
        # Each intersection can set its phase to 0, 1, 2, or 3
        self.intersection_ids = list(self.env.intersections.keys())
        self.action_space = spaces.Discrete(len(PHASES))
        
        # Define observation space
        # Each intersection has incoming lane queues, phase, phase time, speeds
        num_incoming_lanes = max(len(intersection.incoming_lanes) 
                               for intersection in self.env.intersections.values())
        
        # Observation includes queue lengths, phase, phase time, and speeds
        obs_dim = num_incoming_lanes * 2 + 2  # queue + speed for each lane + phase + phase time
        
        self.observation_space = spaces.Box(
            low=0, high=float('inf'), shape=(obs_dim,), dtype=np.float32
        )
    
    def reset(self) -> np.ndarray:
        """
        Reset the environment.
        
        Returns:
            Initial observation
        """
        # Reset the traffic environment
        observations = self.env.reset()
        
        # Return observation for the first intersection (for single-agent)
        return observations[self.intersection_ids[0]]
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Phase to set for the intersection
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        # Convert action to dictionary for the traffic environment
        action_dict = {self.intersection_ids[0]: action}
        
        # Step the environment
        observations, reward, done, info = self.env.step(action_dict)
        
        # Return observation for the first intersection (for single-agent)
        return observations[self.intersection_ids[0]], reward, done, info
    
    def render(self, mode: str = 'human') -> None:
        """
        Render the environment.
        
        Args:
            mode: Rendering mode
        """
        self.env.render(mode)


class DQNTrafficController:
    """
    Deep Q-Network agent for traffic signal control.
    
    This implements a reinforcement learning agent that learns
    optimal traffic light timings.
    """
    
    def __init__(self, state_size: int, action_size: int,
                 learning_rate: float = 0.001, gamma: float = 0.95,
                 epsilon: float = 1.0, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01, batch_size: int = 32,
                 memory_size: int = 2000):
        """
        Initialize the DQN agent.
        
        Args:
            state_size: Size of the state vector
            action_size: Number of possible actions
            learning_rate: Learning rate for the neural network
            gamma: Discount factor for future rewards
            epsilon: Exploration rate (1.0 = always explore)
            epsilon_decay: Rate at which epsilon decreases
            epsilon_min: Minimum exploration rate
            batch_size: Batch size for training
            memory_size: Size of replay memory
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma  # discount rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        
        # Neural network for predicting Q values
        self.model = self._build_model()
        
        # Target network for stability
        self.target_model = self._build_model()
        self.update_target_model()
        
        # Experience replay memory
        self.memory = []
        self.memory_size = memory_size
        
        logger.info(f"Initialized DQN traffic controller with state_size={state_size}, "
                  f"action_size={action_size}")
    
    def _build_model(self) -> keras.Model:
        """
        Build the neural network model for DQN.
        
        Returns:
            Keras Model
        """
        model = keras.Sequential([
            layers.Dense(24, activation='relu', input_shape=(self.state_size,)),
            layers.Dense(24, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_model(self) -> None:
        """Update the target model to match the main model weights."""
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                next_state: np.ndarray, done: bool) -> None:
        """
        Store experience in replay memory.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        # Limit memory size
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)  # Remove oldest memory if full
            
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """
        Choose an action based on the current state.
        
        Uses epsilon-greedy policy during training.
        
        Args:
            state: Current state vector
            training: Whether we're in training mode (True) or evaluation (False)
            
        Returns:
            Selected action
        """
        if training and np.random.rand() <= self.epsilon:
            # Explore - random action
            return random.randrange(self.action_size)
        
        # Exploit - best action from Q-values
        act_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self) -> float:
        """
        Train the model using experience replay.
        
        Returns:
            Loss value from training
        """
        if len(self.memory) < self.batch_size:
            return 0
            
        # Sample a batch from memory
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = np.zeros((self.batch_size, self.state_size))
        targets = np.zeros((self.batch_size, self.action_size))
        
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            states[i] = state
            
            # Calculate target Q-value
            target = self.model.predict(np.expand_dims(state, axis=0), verbose=0)[0]
            
            if done:
                target[action] = reward
            else:
                t = self.target_model.predict(np.expand_dims(next_state, axis=0), verbose=0)[0]
                target[action] = reward + self.gamma * np.amax(t)
                
            targets[i] = target
        
        # Train the model
        history = self.model.fit(states, targets, epochs=1, verbose=0)
        loss = history.history['loss'][0]
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return loss
    
    def load(self, filepath: str) -> None:
        """Load model weights from file."""
        self.model.load_weights(filepath)
        self.update_target_model()
        logger.info(f"Loaded model from {filepath}")
    
    def save(self, filepath: str) -> None:
        """Save model weights to file."""
        self.model.save_weights(filepath)
        logger.info(f"Saved model to {filepath}")


class TrafficSignalTrainer:
    """
    Class for training and evaluating traffic signal controllers.
    """
    
    def __init__(self, config_path: str = None, model_dir: str = "./models", 
                 log_dir: str = "./logs"):
        """
        Initialize the trainer.
        
        Args:
            config_path: Path to configuration file
            model_dir: Directory for saving models
            log_dir: Directory for saving logs
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Create directories
        self.model_dir = model_dir
        self.log_dir = log_dir
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Create environment and agent
        self.env = TrafficSignalGymEnv(self.config)
        
        # Create agent
        self.agent = DQNTrafficController(
            state_size=self.env.observation_space.shape[0],
            action_size=self.env.action_space.n,
            **self.config.get('agent', {})
        )
        
        # Training metrics
        self.training_rewards = []
        self.training_throughputs = []
        self.training_queue_lengths = []
        self.training_travel_times = []
        self.eval_metrics = []
        
        logger.info(f"Traffic signal trainer initialized with config: {self.config}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        # Default configuration
        default_config = {
            'duration': 3600,  # 1 hour simulation
            'timestep': 1.0,
            'max_vehicles': 1000,
            'spawn_rate': 0.1,
            'seed': 42,
            'agent': {
                'learning_rate': 0.001,
                'gamma': 0.95,
                'epsilon': 1.0,
                'epsilon_decay': 0.995,
                'epsilon_min': 0.01,
                'batch_size': 32,
                'memory_size': 2000
            }
        }
        
        # Load from file if provided
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    # Update default with loaded config
                    default_config.update(config)
            except Exception as e:
                logger.error(f"Error loading config from {config_path}: {e}")
                logger.info("Using default configuration")
        
        return default_config
    
    def train(self, episodes: int = 100, target_update_freq: int = 10,
             evaluate_freq: int = 10, save_freq: int = 25) -> None:
        """
        Train the agent on the traffic control task.
        
        Args:
            episodes: Number of episodes to train for
            target_update_freq: How often to update target network
            evaluate_freq: How often to evaluate performance
            save_freq: How often to save the model
        """
        logger.info(f"Starting training for {episodes} episodes")
        
        # Setup TensorBoard if available
        try:
            current_time = time.strftime("%Y%m%d-%H%M%S")
            train_log_dir = os.path.join(self.log_dir, f'tensorboard/dqn_{current_time}')
            summary_writer = tf.summary.create_file_writer(train_log_dir)
        except:
            summary_writer = None
            logger.warning("TensorBoard not available, skipping logging")
        
        # Training loop
        for episode in range(1, episodes + 1):
            # Reset environment
            state = self.env.reset()
            episode_reward = 0
            episode_throughput = 0
            episode_queues = []
            done = False
            step = 0
            
            # Episode loop
            while not done:
                # Choose action
                action = self.agent.act(state)
                
                # Take action
                next_state, reward, done, info = self.env.step(action)
                
                # Store transition
                self.agent.remember(state, action, reward, next_state, done)
                
                # Update state and stats
                state = next_state
                episode_reward += reward
                episode_throughput += info['throughput']
                episode_queues.append(info['queues'])
                step += 1
                
                # Train the agent
                loss = self.agent.replay()
                
                # Update target model periodically
                if step % target_update_freq == 0:
                    self.agent.update_target_model()
            
            # Get final metrics for the episode
            metrics = self.env.env.get_traffic_metrics()
            
            # Store metrics
            self.training_rewards.append(episode_reward)
            self.training_throughputs.append(episode_throughput)
            self.training_queue_lengths.append(np.mean([sum(q.values()) for q in episode_queues]))
            self.training_travel_times.append(metrics['avg_travel_time'])
            
            # Log to TensorBoard
            if summary_writer:
                with summary_writer.as_default():
                    tf.summary.scalar('episode_reward', episode_reward, step=episode)
                    tf.summary.scalar('throughput', episode_throughput, step=episode)
                    tf.summary.scalar('avg_queue', np.mean([sum(q.values()) for q in episode_queues]), step=episode)
                    tf.summary.scalar('avg_travel_time', metrics['avg_travel_time'], step=episode)
                    tf.summary.scalar('epsilon', self.agent.epsilon, step=episode)
            
            # Print progress
            logger.info(f"Episode: {episode}/{episodes}, Reward: {episode_reward:.2f}, "
                      f"Throughput: {episode_throughput}, Avg Travel Time: {metrics['avg_travel_time']:.2f}s, "
                      f"Epsilon: {self.agent.epsilon:.4f}")
            
            # Evaluate periodically
            if episode % evaluate_freq == 0:
                eval_metrics = self.evaluate(5)
                self.eval_metrics.append(eval_metrics)
                
                logger.info(f"Evaluation - Avg Reward: {eval_metrics['avg_reward']:.2f}, "
                          f"Avg Travel Time: {eval_metrics['avg_travel_time']:.2f}s")
                
                # Log evaluation metrics
                if summary_writer:
                    with summary_writer.as_default():
                        tf.summary.scalar('eval_reward', eval_metrics['avg_reward'], step=episode)
                        tf.summary.scalar('eval_travel_time', eval_metrics['avg_travel_time'], step=episode)
            
            # Save model periodically
            if episode % save_freq == 0:
                self.agent.save(os.path.join(self.model_dir, f"traffic_dqn_ep{episode}.h5"))
                
                # Save learning curves
                self.visualize_learning_curves(
                    save_path=os.path.join(self.log_dir, f"learning_curves_ep{episode}.png")
                )
        
        # Final save
        self.agent.save(os.path.join(self.model_dir, "traffic_dqn_final.h5"))
        logger.info("Training completed")
        
        # Save final learning curves
        self.visualize_learning_curves(
            save_path=os.path.join(self.log_dir, "learning_curves_final.png")
        )
    
    def evaluate(self, episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate the agent's performance.
        
        Args:
            episodes: Number of episodes to evaluate on
            
        Returns:
            Dictionary of evaluation metrics
        """
        rewards = []
        throughputs = []
        travel_times = []
        queue_lengths = []
        
        # Run evaluation episodes
        for _ in range(episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_throughput = 0
            episode_queues = []
            done = False
            
            while not done:
                # Choose action (no exploration)
                action = self.agent.act(state, training=False)
                next_state, reward, done, info = self.env.step(action)
                
                state = next_state
                episode_reward += reward
                episode_throughput += info['throughput']
                episode_queues.append(info['queues'])
            
            # Get final metrics
            metrics = self.env.env.get_traffic_metrics()
            
            rewards.append(episode_reward)
            throughputs.append(episode_throughput)
            travel_times.append(metrics['avg_travel_time'])
            queue_lengths.append(np.mean([sum(q.values()) for q in episode_queues]))
        
        # Calculate averages
        eval_metrics = {
            'avg_reward': np.mean(rewards),
            'avg_throughput': np.mean(throughputs),
            'avg_travel_time': np.mean(travel_times),
            'avg_queue': np.mean(queue_lengths)
        }
        
        return eval_metrics
    
    def visualize_learning_curves(self, save_path: str = None) -> None:
        """
        Visualize the learning curves.
        
        Args:
            save_path: Optional path to save the figure
        """
        if not self.training_rewards:
            logger.warning("No training data available for visualization")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot rewards
        axes[0, 0].plot(self.training_rewards)
        axes[0, 0].set_title('Episode Reward')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True)
        
        # Plot throughput
        axes[0, 1].plot(self.training_throughputs)
        axes[0, 1].set_title('Episode Throughput')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Vehicles')
        axes[0, 1].grid(True)
        
        # Plot queue lengths
        axes[1, 0].plot(self.training_queue_lengths)
        axes[1, 0].set_title('Average Queue Length')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Vehicles')
        axes[1, 0].grid(True)
        
        # Plot travel times
        axes[1, 1].plot(self.training_travel_times)
        axes[1, 1].set_title('Average Travel Time')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Seconds')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            logger.info(f"Learning curves saved to {save_path}")
            plt.close()
        else:
            plt.show()
    
    def compare_with_fixed_timing(self, episodes: int = 10) -> Dict[str, Dict[str, float]]:
        """
        Compare the RL agent with fixed-time control.
        
        Args:
            episodes: Number of episodes for comparison
            
        Returns:
            Dictionary of metrics for both controllers
        """
        # Metrics for RL controller
        rl_metrics = self.evaluate(episodes)
        
        # Create a new environment for fixed-time control
        fixed_env = TrafficSignalGymEnv(self.config)
        
        # Run fixed timing strategy
        fixed_rewards = []
        fixed_throughputs = []
        fixed_travel_times = []
        fixed_queue_lengths = []
        
        for _ in range(episodes):
            state = fixed_env.reset()
            episode_reward = 0
            episode_throughput = 0
            episode_queues = []
            done = False
            step = 0
            
            while not done:
                # Fixed timing strategy - cycle through phases
                # Simplistic fixed timing: change phase every 30 seconds
                phase = (step // 30) % len(PHASES)
                next_state, reward, done, info = fixed_env.step(phase)
                
                state = next_state
                episode_reward += reward
                episode_throughput += info['throughput']
                episode_queues.append(info['queues'])
                step += 1
            
            # Get final metrics
            metrics = fixed_env.env.get_traffic_metrics()
            
            fixed_rewards.append(episode_reward)
            fixed_throughputs.append(episode_throughput)
            fixed_travel_times.append(metrics['avg_travel_time'])
            fixed_queue_lengths.append(np.mean([sum(q.values()) for q in episode_queues]))
        
        # Calculate averages for fixed timing
        fixed_metrics = {
            'avg_reward': np.mean(fixed_rewards),
            'avg_throughput': np.mean(fixed_throughputs),
            'avg_travel_time': np.mean(fixed_travel_times),
            'avg_queue': np.mean(fixed_queue_lengths)
        }
        
        # Calculate improvements
        improvements = {
            'reward_improvement': (rl_metrics['avg_reward'] - fixed_metrics['avg_reward']) / abs(fixed_metrics['avg_reward']) * 100,
            'throughput_improvement': (rl_metrics['avg_throughput'] - fixed_metrics['avg_throughput']) / fixed_metrics['avg_throughput'] * 100,
            'travel_time_improvement': (fixed_metrics['avg_travel_time'] - rl_metrics['avg_travel_time']) / fixed_metrics['avg_travel_time'] * 100,
            'queue_improvement': (fixed_metrics['avg_queue'] - rl_metrics['avg_queue']) / fixed_metrics['avg_queue'] * 100
        }
        
        # Print comparison
        logger.info("\n=== RL vs Fixed Timing Comparison ===")
        logger.info(f"Average Reward: RL = {rl_metrics['avg_reward']:.2f}, Fixed = {fixed_metrics['avg_reward']:.2f}, "
                  f"Improvement = {improvements['reward_improvement']:.1f}%")
        logger.info(f"Average Throughput: RL = {rl_metrics['avg_throughput']:.2f}, Fixed = {fixed_metrics['avg_throughput']:.2f}, "
                  f"Improvement = {improvements['throughput_improvement']:.1f}%")
        logger.info(f"Average Travel Time: RL = {rl_metrics['avg_travel_time']:.2f}s, Fixed = {fixed_metrics['avg_travel_time']:.2f}s, "
                  f"Improvement = {improvements['travel_time_improvement']:.1f}%")
        logger.info(f"Average Queue Length: RL = {rl_metrics['avg_queue']:.2f}, Fixed = {fixed_metrics['avg_queue']:.2f}, "
                  f"Improvement = {improvements['queue_improvement']:.1f}%")
        
        return {
            'rl': rl_metrics,
            'fixed': fixed_metrics,
            'improvements': improvements
        }
    
    def visualize_comparison(self, comparison_data: Dict[str, Dict[str, float]],
                           save_path: str = None) -> None:
        """
        Visualize the comparison between RL and fixed-time control.
        
        Args:
            comparison_data: Results from compare_with_fixed_timing
            save_path: Optional path to save the figure
        """
        metrics = ['avg_reward', 'avg_throughput', 'avg_travel_time', 'avg_queue']
        metric_names = ['Average Reward', 'Average Throughput', 'Average Travel Time (s)', 'Average Queue Length']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            ax = axes[i]
            rl_value = comparison_data['rl'][metric]
            fixed_value = comparison_data['fixed'][metric]
            
            # Better values are higher for rewards and throughput, lower for travel time and queue
            if metric in ['avg_travel_time', 'avg_queue']:
                better = min(rl_value, fixed_value)
                worse = max(rl_value, fixed_value)
                if better == rl_value:
                    colors = ['green', 'red']
                else:
                    colors = ['red', 'green']
            else:
                better = max(rl_value, fixed_value)
                worse = min(rl_value, fixed_value)
                if better == rl_value:
                    colors = ['green', 'red']
                else:
                    colors = ['red', 'green']
            
            ax.bar(['RL Controller', 'Fixed Timing'], [rl_value, fixed_value], color=colors)
            ax.set_title(name)
            ax.grid(axis='y')
            
            # Add value labels
            for j, v in enumerate([rl_value, fixed_value]):
                ax.text(j, v * 1.01, f"{v:.2f}", ha='center')
                
            # Add improvement text
            imp_key = metric + '_improvement'
            if imp_key in comparison_data['improvements']:
                imp_value = comparison_data['improvements'][imp_key]
                if imp_value > 0:
                    imp_text = f"+{imp_value:.1f}%"
                else:
                    imp_text = f"{imp_value:.1f}%"
                ax.set_xlabel(f"Improvement: {imp_text}")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            logger.info(f"Comparison visualization saved to {save_path}")
            plt.close()
        else:
            plt.show()


def run_traffic_signal_ai(mode: str = 'train', config_path: str = None, model_path: str = None,
                        episodes: int = 100, runs: int = 10):
    """
    Main function to run the traffic signal AI system.
    
    Args:
        mode: 'train', 'evaluate', 'compare', or 'visualize'
        config_path: Path to configuration file
        model_path: Path to saved model (for evaluate/compare modes)
        episodes: Number of episodes for training or comparison
        runs: Number of evaluation runs
    """
    # Create the trainer
    trainer = TrafficSignalTrainer(config_path)
    
    # Run in specified mode
    if mode == 'train':
        trainer.train(episodes=episodes)
        
    elif mode == 'evaluate':
        if model_path:
            # Load the model
            trainer.agent.load(model_path)
            
        # Run evaluation
        metrics = trainer.evaluate(runs)
        
        print("\n=== Evaluation Results ===")
        print(f"Average Reward: {metrics['avg_reward']:.2f}")
        print(f"Average Throughput: {metrics['avg_throughput']:.2f}")
        print(f"Average Travel Time: {metrics['avg_travel_time']:.2f}s")
        print(f"Average Queue Length: {metrics['avg_queue']:.2f}")
        
    elif mode == 'compare':
        if model_path:
            # Load the model
            trainer.agent.load(model_path)
            
        # Compare with fixed timing
        comparison = trainer.compare_with_fixed_timing(runs)
        
        # Visualize comparison
        trainer.visualize_comparison(
            comparison,
            save_path=os.path.join(trainer.log_dir, "rl_vs_fixed_comparison.png")
        )
        
    elif mode == 'visualize':
        if model_path:
            # Load the model
            trainer.agent.load(model_path)
            
        # Run a single episode with visualization
        state = trainer.env.reset()
        done = False
        
        while not done:
            # Choose action (no exploration)
            action = trainer.agent.act(state, training=False)
            next_state, reward, done, info = trainer.env.step(action)
            
            # Render
            trainer.env.render()
            time.sleep(0.1)  # Slow down for better visibility
            
            state = next_state
            
        # Show final metrics
        metrics = trainer.env.env.get_traffic_metrics()
        
        print("\n=== Final Traffic Metrics ===")
        print(f"Completed Trips: {metrics['completed_trips']}")
        print(f"Average Travel Time: {metrics['avg_travel_time']:.2f}s")
        print(f"Average Wait Time: {metrics['avg_wait_time']:.2f}s")
        print(f"Maximum Queue Length: {metrics['max_queue']}")
    
    else:
        logger.error(f"Unknown mode: {mode}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Traffic Signal AI Controller')
    parser.add_argument('--mode', type=str, default='train', 
                      choices=['train', 'evaluate', 'compare', 'visualize'],
                      help='Mode to run the system in')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--model', type=str, help='Path to saved model')
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes for training')
    parser.add_argument('--runs', type=int, default=10, help='Number of evaluation runs')
    
    args = parser.parse_args()
    
    run_traffic_signal_ai(
        mode=args.mode,
        config_path=args.config,
        model_path=args.model,
        episodes=args.episodes,
        runs=args.runs
    )

"""
SUMMARY:
========
This module implements an AI-based traffic signal control system using reinforcement learning.
The system creates a simulation of traffic flow at intersections and trains an RL agent to 
optimize traffic signal timing for minimal congestion and wait times.

Key components:
1. Vehicle - Simulated vehicles with realistic physics
2. Lane - Road segments that contain vehicles
3. Intersection - Traffic intersections with signal control
4. TrafficEnvironment - Simulation environment for traffic flow
5. DQNTrafficController - Deep Q-Network agent for signal control
6. TrafficSignalTrainer - Training and evaluation framework

The system compares the AI controller with traditional fixed-timing approaches and
visualizes the improvements in traffic flow metrics.

TODO:
=====
1. Add support for connected vehicle data integration (V2I communication)
2. Implement multi-intersection coordination using multi-agent RL
3. Add pedestrian crossing simulation and priority
4. Integrate with real traffic camera feeds for online learning
5. Add support for emergency vehicle preemption
6. Implement more sophisticated traffic demand patterns (rush hour, events)
7. Add weather effects on traffic flow and visibility
8. Create a more realistic vehicle routing model with GPS-like navigation
9. Implement a GUI for better visualization of traffic flows
10. Add support for adaptive phase sequences, not just timing
11. Create an A/B testing framework for comparing multiple control strategies
12. Integrate with city traffic management systems via standardized APIs
13. Add bicycle lanes and specialized signal phases for cycling traffic
"""
