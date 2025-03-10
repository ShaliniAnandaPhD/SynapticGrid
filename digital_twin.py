"""
Generates a virtual model of the city to test AI optimizations before real-world
deployment. This API allows other smart city components to interact with a
simulated version of the city infrastructure.
"""

import numpy as np
import pandas as pd
import networkx as nx
import geopandas as gpd
import json
import time
import logging
import os
import pickle
import threading
import queue
import uuid
from typing import Dict, List, Tuple, Any, Optional, Union
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from shapely.geometry import Point, LineString, Polygon
import uvicorn
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("digital_twin.log"), logging.StreamHandler()]
)
logger = logging.getLogger("DigitalTwin")

# Constants
DEFAULT_TIME_STEP = 60  # seconds per simulation step
MAX_SIMULATION_DURATION = 86400  # maximum simulation duration (24 hours)
DEFAULT_CITY_BOUNDS = (37.7, -122.5, 37.8, -122.4)  # Default city bounds (SF)


# Data models for API
class SimulationConfig(BaseModel):
    """Configuration for a simulation run."""
    duration: int = Field(3600, description="Simulation duration in seconds")
    time_step: int = Field(60, description="Simulation time step in seconds")
    start_time: Optional[str] = Field(None, description="Simulation start time (ISO format)")
    traffic_scale: float = Field(1.0, description="Traffic volume scaling factor")
    random_seed: Optional[int] = Field(None, description="Random seed for simulation")
    scenarios: List[Dict[str, Any]] = Field([], description="List of scenario configurations")


class SimulationStatus(BaseModel):
    """Status of a simulation run."""
    id: str
    status: str  # "running", "completed", "failed"
    progress: float  # 0-100
    current_time: Optional[str] = None
    message: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None


class ComponentConfig(BaseModel):
    """Configuration for a city component."""
    component_type: str
    component_id: str
    properties: Dict[str, Any] = {}
    location: Optional[Dict[str, float]] = None
    connections: List[str] = []


class ComponentState(BaseModel):
    """State update for a component."""
    component_id: str
    state: Dict[str, Any]
    timestamp: Optional[str] = None


class QueryParams(BaseModel):
    """Parameters for querying simulation results."""
    component_types: Optional[List[str]] = None
    component_ids: Optional[List[str]] = None
    time_range: Optional[Tuple[str, str]] = None
    metrics: Optional[List[str]] = None
    aggregation: Optional[str] = None


# Core classes
class CityComponent:
    """
    Base class for all city components in the digital twin.
    
    All specific components (traffic lights, buildings, etc.) inherit from this.
    """
    
    def __init__(self, component_id: str, component_type: str, properties: Dict = None,
                location: Dict = None):
        """
        Initialize a city component.
        
        Args:
            component_id: Unique identifier
            component_type: Type of component
            properties: Component properties
            location: Geo location (lat/lon)
        """
        self.id = component_id
        self.type = component_type
        self.properties = properties or {}
        self.location = location
        self.state = {}
        self.state_history = []
        self.connections = []  # IDs of connected components
        
    def update(self, time_step: float, environment_state: Dict) -> Dict:
        """
        Update the component state for one time step.
        
        Args:
            time_step: Simulation time step in seconds
            environment_state: State of surrounding environment
            
        Returns:
            Updated component state
        """
        # Base implementation does nothing
        # Subclasses should override this method
        return self.state
    
    def connect_to(self, component_id: str) -> None:
        """
        Connect this component to another component.
        
        Args:
            component_id: ID of the component to connect to
        """
        if component_id not in self.connections:
            self.connections.append(component_id)
    
    def save_state(self, timestamp: float) -> None:
        """
        Save the current state to history.
        
        Args:
            timestamp: Current simulation time
        """
        history_entry = {
            'timestamp': timestamp,
            'state': self.state.copy()
        }
        self.state_history.append(history_entry)
        
        # Limit history size to prevent memory issues
        if len(self.state_history) > 1000:
            self.state_history = self.state_history[-1000:]
    
    def get_state_at(self, timestamp: float) -> Dict:
        """
        Get component state at a specific time.
        
        Args:
            timestamp: Time to get state for
            
        Returns:
            Component state at requested time
        """
        # Find the closest state entry
        if not self.state_history:
            return self.state
            
        closest_entry = min(self.state_history, 
                          key=lambda entry: abs(entry['timestamp'] - timestamp))
                          
        return closest_entry['state']
    
    def to_dict(self) -> Dict:
        """
        Convert component to dictionary representation.
        
        Returns:
            Dictionary with component attributes
        """
        return {
            'id': self.id,
            'type': self.type,
            'properties': self.properties,
            'location': self.location,
            'state': self.state,
            'connections': self.connections
        }


class TrafficSignal(CityComponent):
    """Traffic signal component with phases and timing."""
    
    def __init__(self, component_id: str, properties: Dict = None, location: Dict = None):
        """Initialize traffic signal with default phases."""
        super().__init__(component_id, "traffic_signal", properties, location)
        
        # Default properties if not provided
        if not self.properties.get('phases'):
            self.properties['phases'] = [
                {'id': 0, 'name': 'North-South Green', 'duration': 30},
                {'id': 1, 'name': 'North-South Yellow', 'duration': 3},
                {'id': 2, 'name': 'East-West Green', 'duration': 30},
                {'id': 3, 'name': 'East-West Yellow', 'duration': 3}
            ]
        
        # Initial state
        self.state = {
            'current_phase': 0,
            'phase_time': 0,
            'is_active': True
        }
    
    def update(self, time_step: float, environment_state: Dict) -> Dict:
        """Update traffic signal state based on phases and timing."""
        if not self.state['is_active']:
            return self.state
            
        # Update phase time
        self.state['phase_time'] += time_step
        
        # Check if time to change phase
        current_phase = self.state['current_phase']
        phase_duration = self.properties['phases'][current_phase]['duration']
        
        if self.state['phase_time'] >= phase_duration:
            # Advance to next phase
            next_phase = (current_phase + 1) % len(self.properties['phases'])
            self.state['current_phase'] = next_phase
            self.state['phase_time'] = 0
            
            # If an AI control system is connected, it might override this phase
            if environment_state.get('ai_control'):
                ai_recommendation = environment_state.get('recommended_phase')
                if ai_recommendation is not None:
                    self.state['current_phase'] = ai_recommendation
        
        return self.state


class Building(CityComponent):
    """Building component with occupancy and energy usage."""
    
    def __init__(self, component_id: str, properties: Dict = None, location: Dict = None):
        """Initialize building with occupancy and energy properties."""
        super().__init__(component_id, "building", properties, location)
        
        # Default properties if not provided
        if 'max_occupancy' not in self.properties:
            self.properties['max_occupancy'] = 100
        if 'energy_efficiency' not in self.properties:
            self.properties['energy_efficiency'] = 0.7  # 0-1 scale
        if 'type' not in self.properties:
            self.properties['type'] = 'commercial'  # commercial, residential, etc.
        
        # Initial state
        self.state = {
            'current_occupancy': 0,
            'energy_usage_kw': 0,
            'temperature': 22,  # Celsius
            'hvac_active': False
        }
    
    def update(self, time_step: float, environment_state: Dict) -> Dict:
        """Update building state based on time of day and environment."""
        # Get time of day from environment
        hour = environment_state.get('hour', 12)
        day_type = environment_state.get('day_type', 'weekday')  # weekday or weekend
        outside_temp = environment_state.get('temperature', 20)  # Celsius
        
        # Update occupancy based on time patterns
        if self.properties['type'] == 'commercial':
            if day_type == 'weekday':
                if 9 <= hour < 17:  # Business hours
                    target_occupancy = self.properties['max_occupancy'] * 0.8
                elif 7 <= hour < 9 or 17 <= hour < 19:  # Transition hours
                    target_occupancy = self.properties['max_occupancy'] * 0.5
                else:  # Off hours
                    target_occupancy = self.properties['max_occupancy'] * 0.1
            else:  # Weekend
                target_occupancy = self.properties['max_occupancy'] * 0.2
        else:  # Residential
            if day_type == 'weekday':
                if 7 <= hour < 9 or 17 <= hour < 23:  # Home hours
                    target_occupancy = self.properties['max_occupancy'] * 0.8
                elif 9 <= hour < 17:  # Work hours
                    target_occupancy = self.properties['max_occupancy'] * 0.3
                else:  # Sleep hours
                    target_occupancy = self.properties['max_occupancy'] * 0.9
            else:  # Weekend
                if 10 <= hour < 23:  # Awake hours
                    target_occupancy = self.properties['max_occupancy'] * 0.7
                else:  # Sleep hours
                    target_occupancy = self.properties['max_occupancy'] * 0.9
        
        # Gradually adjust current occupancy towards target
        adjustment_rate = 0.1 * time_step / 60  # 10% adjustment per minute
        occupancy_diff = target_occupancy - self.state['current_occupancy']
        self.state['current_occupancy'] += occupancy_diff * adjustment_rate
        
        # Calculate energy usage
        base_energy = 2.0  # Base energy in kW
        occupancy_factor = self.state['current_occupancy'] / self.properties['max_occupancy']
        
        # HVAC energy depends on temperature difference
        temp_diff = abs(self.state['temperature'] - outside_temp)
        hvac_needed = temp_diff > 3  # Only active if difference > 3°C
        
        if hvac_needed:
            self.state['hvac_active'] = True
            hvac_energy = 5.0 + temp_diff * 0.5  # Base + scaling with temp difference
        else:
            self.state['hvac_active'] = False
            hvac_energy = 0
            
        # Apply efficiency factor
        efficiency = self.properties['energy_efficiency']
        self.state['energy_usage_kw'] = (base_energy + 
                                       occupancy_factor * 10 + 
                                       hvac_energy) / efficiency
        
        # Update temperature
        if self.state['hvac_active']:
            # Temperature moves toward desired temperature (22°C)
            temp_adjustment = (22 - self.state['temperature']) * 0.1 * time_step / 60
            self.state['temperature'] += temp_adjustment
        else:
            # Temperature moves toward outside temperature
            temp_adjustment = (outside_temp - self.state['temperature']) * 0.05 * time_step / 60
            self.state['temperature'] += temp_adjustment
        
        return self.state


class Road(CityComponent):
    """Road segment with traffic state and capacity."""
    
    def __init__(self, component_id: str, properties: Dict = None, location: Dict = None):
        """Initialize road with traffic properties."""
        super().__init__(component_id, "road", properties, location)
        
        # Default properties if not provided
        if 'length' not in self.properties:
            self.properties['length'] = 1000  # meters
        if 'lanes' not in self.properties:
            self.properties['lanes'] = 2
        if 'max_speed' not in self.properties:
            self.properties['max_speed'] = 50  # km/h
        if 'capacity' not in self.properties:
            # Default capacity in vehicles per hour per lane
            self.properties['capacity'] = 1800 * self.properties['lanes']
        
        # Initial state
        self.state = {
            'vehicle_count': 0,
            'avg_speed': self.properties['max_speed'],
            'congestion': 0.0,  # 0-1 scale
            'travel_time': self.properties['length'] / (self.properties['max_speed'] / 3.6)  # seconds
        }
    
    def update(self, time_step: float, environment_state: Dict) -> Dict:
        """Update road state based on traffic patterns and connected roads."""
        # Get time of day and traffic scale from environment
        hour = environment_state.get('hour', 12)
        traffic_scale = environment_state.get('traffic_scale', 1.0)
        
        # Time-based traffic patterns
        time_factors = {
            0: 0.1,  # Midnight
            3: 0.05, # 3 AM
            6: 0.3,  # 6 AM
            8: 1.0,  # 8 AM (rush hour)
            10: 0.6, # 10 AM
            12: 0.7, # Noon
            15: 0.6, # 3 PM
            17: 1.0, # 5 PM (rush hour)
            19: 0.7, # 7 PM
            22: 0.3  # 10 PM
        }
        
        # Get the closest hour factors and interpolate
        hours = sorted(time_factors.keys())
        lower_hour = max([h for h in hours if h <= hour], default=hours[0])
        upper_hour = min([h for h in hours if h >= hour], default=hours[-1])
        
        if lower_hour == upper_hour:
            time_factor = time_factors[lower_hour]
        else:
            # Linear interpolation
            weight = (hour - lower_hour) / (upper_hour - lower_hour)
            time_factor = (1 - weight) * time_factors[lower_hour] + weight * time_factors[upper_hour]
        
        # Calculate target vehicle count based on capacity, time factor, and global scale
        capacity = self.properties['capacity']
        target_count = int(capacity * time_factor * traffic_scale * (time_step / 3600))
        
        # Calculate inflow from connected roads
        inflow = 0
        connected_roads = environment_state.get('connected_roads', {})
        for road_id in self.connections:
            if road_id in connected_roads:
                # Simplified flow model based on outflow from connected roads
                road_state = connected_roads[road_id]
                outflow_pct = 0.3  # Assume 30% of traffic flows to this road
                inflow += int(road_state.get('vehicle_count', 0) * outflow_pct * (time_step / 3600))
        
        # Add random variation (±20%)
        random_factor = 1.0 + 0.2 * (np.random.random() - 0.5)
        target_count = int(target_count * random_factor) + inflow
        
        # Gradually adjust vehicle count (not instant)
        count_diff = target_count - self.state['vehicle_count']
        adjustment_rate = 0.2 * time_step / 60  # 20% adjustment per minute
        self.state['vehicle_count'] += int(count_diff * adjustment_rate)
        self.state['vehicle_count'] = max(0, self.state['vehicle_count'])  # Ensure non-negative
        
        # Calculate congestion level
        hourly_volume = self.state['vehicle_count'] * (3600 / time_step)
        congestion_ratio = min(1.0, hourly_volume / self.properties['capacity'])
        self.state['congestion'] = congestion_ratio
        
        # Calculate average speed based on congestion
        # Using BPR (Bureau of Public Roads) formula
        free_flow_speed = self.properties['max_speed']
        alpha, beta = 0.15, 4.0  # BPR parameters
        speed_reduction = 1.0 / (1.0 + alpha * (congestion_ratio ** beta))
        self.state['avg_speed'] = free_flow_speed * speed_reduction
        
        # Calculate travel time
        self.state['travel_time'] = self.properties['length'] / (self.state['avg_speed'] / 3.6)  # seconds
        
        return self.state


class Scenario:
    """
    Represents a scenario that can be applied to the simulation.
    
    Scenarios are events or conditions that affect the normal operation
    of components, such as accidents, weather events, or special conditions.
    """
    
    def __init__(self, scenario_id: str, scenario_type: str, properties: Dict = None):
        """
        Initialize a scenario.
        
        Args:
            scenario_id: Unique identifier
            scenario_type: Type of scenario
            properties: Scenario properties
        """
        self.id = scenario_id
        self.type = scenario_type
        self.properties = properties or {}
        self.active = False
        self.start_time = None
        self.end_time = None
    
    def activate(self, time: float) -> None:
        """
        Activate the scenario at the specified time.
        
        Args:
            time: Activation time
        """
        self.active = True
        self.start_time = time
    
    def deactivate(self, time: float) -> None:
        """
        Deactivate the scenario at the specified time.
        
        Args:
            time: Deactivation time
        """
        self.active = False
        self.end_time = time
    
    def apply_effects(self, components: Dict[str, CityComponent], time: float) -> None:
        """
        Apply scenario effects to components.
        
        Args:
            components: Dictionary of city components
            time: Current simulation time
        """
        if not self.active:
            return
            
        # Different effects based on scenario type
        if self.type == "traffic_accident":
            self._apply_accident_effects(components, time)
        elif self.type == "weather_event":
            self._apply_weather_effects(components, time)
        elif self.type == "power_outage":
            self._apply_power_outage_effects(components, time)
        elif self.type == "special_event":
            self._apply_special_event_effects(components, time)
    
    def _apply_accident_effects(self, components: Dict[str, CityComponent], time: float) -> None:
        """Apply effects of a traffic accident."""
        # Get affected road
        road_id = self.properties.get('road_id')
        if not road_id or road_id not in components:
            return
            
        road = components[road_id]
        if road.type != 'road':
            return
            
        # Reduce capacity and speed
        severity = self.properties.get('severity', 0.5)  # 0-1 scale
        
        # Modify road state
        capacity_reduction = 0.3 + severity * 0.6  # 30-90% reduction
        effective_capacity = road.properties['capacity'] * (1 - capacity_reduction)
        
        speed_reduction = 0.4 + severity * 0.5  # 40-90% reduction
        effective_speed = road.properties['max_speed'] * (1 - speed_reduction)
        
        # Apply modifications to road state
        road.state['congestion'] = min(1.0, road.state['congestion'] + severity * 0.5)
        road.state['avg_speed'] = effective_speed
        
        # Update travel time based on reduced speed
        road.state['travel_time'] = road.properties['length'] / (effective_speed / 3.6)
        
        # Affect connected roads too
        for connected_id in road.connections:
            if connected_id in components and components[connected_id].type == 'road':
                connected_road = components[connected_id]
                connected_road.state['congestion'] = min(1.0, connected_road.state['congestion'] + severity * 0.3)
                
                # Reduce speed on connected roads too but less severely
                connected_speed = connected_road.state['avg_speed'] * (1 - speed_reduction * 0.5)
                connected_road.state['avg_speed'] = connected_speed
                connected_road.state['travel_time'] = connected_road.properties['length'] / (connected_speed / 3.6)
    
    def _apply_weather_effects(self, components: Dict[str, CityComponent], time: float) -> None:
        """Apply effects of a weather event."""
        weather_type = self.properties.get('weather_type', 'rain')
        severity = self.properties.get('severity', 0.5)
        affected_area = self.properties.get('area', 'all')
        
        # Define impact factors for different weather types
        impact_factors = {
            'rain': {
                'road_speed': 0.9,  # 10% speed reduction
                'road_capacity': 0.95,  # 5% capacity reduction
                'building_energy': 1.1  # 10% energy increase
            },
            'snow': {
                'road_speed': 0.6,  # 40% speed reduction
                'road_capacity': 0.7,  # 30% capacity reduction
                'building_energy': 1.3  # 30% energy increase
            },
            'storm': {
                'road_speed': 0.7,  # 30% speed reduction
                'road_capacity': 0.8,  # 20% capacity reduction
                'building_energy': 1.2  # 20% energy increase
            },
            'heat_wave': {
                'road_speed': 0.95,  # 5% speed reduction
                'road_capacity': 1.0,  # No capacity reduction
                'building_energy': 1.5  # 50% energy increase
            },
            'fog': {
                'road_speed': 0.7,  # 30% speed reduction
                'road_capacity': 0.9,  # 10% capacity reduction
                'building_energy': 1.05  # 5% energy increase
            }
        }
        
        if weather_type not in impact_factors:
            weather_type = 'rain'  # Default to rain if unknown
        
        impact = impact_factors[weather_type]
        
        # Scale impact by severity
        for key in impact:
            if impact[key] < 1.0:
                # For reductions, more severe = lower value
                impact[key] = 1.0 - (1.0 - impact[key]) * severity
            else:
                # For increases, more severe = higher value
                impact[key] = 1.0 + (impact[key] - 1.0) * severity
        
        # Apply effects to all components in the affected area
        for component_id, component in components.items():
            # Check if component is in affected area
            if affected_area != 'all':
                # Would need to check component location against area polygon
                # Simplified check for now
                if component.location and 'area' in component.location:
                    if component.location['area'] != affected_area:
                        continue
            
            # Apply effects based on component type
            if component.type == 'road':
                component.state['avg_speed'] *= impact['road_speed']
                # Update travel time based on new speed
                component.state['travel_time'] = component.properties['length'] / (component.state['avg_speed'] / 3.6)
                
                # Weather increases congestion
                congestion_increase = (1.0 - impact['road_capacity']) * 0.5
                component.state['congestion'] = min(1.0, component.state['congestion'] + congestion_increase)
                
            elif component.type == 'building':
                # Weather affects energy usage
                component.state['energy_usage_kw'] *= impact['building_energy']
                
                # Weather might affect temperature
                if weather_type == 'heat_wave':
                    # Increase outside temperature
                    outside_temp_increase = 5.0 * severity
                    component.state['temperature'] += outside_temp_increase * 0.2  # Building temp rises slowly
                elif weather_type in ['snow', 'rain']:
                    # Decrease outside temperature
                    outside_temp_decrease = 3.0 * severity
                    component.state['temperature'] -= outside_temp_decrease * 0.1  # Building temp drops slowly
    
    def _apply_power_outage_effects(self, components: Dict[str, CityComponent], time: float) -> None:
        """Apply effects of a power outage."""
        affected_area = self.properties.get('area', 'all')
        outage_type = self.properties.get('outage_type', 'blackout')  # blackout or brownout
        
        # Apply effects to all components in the affected area
        for component_id, component in components.items():
            # Check if component is in affected area
            if affected_area != 'all':
                if component.location and 'area' in component.location:
                    if component.location['area'] != affected_area:
                        continue
            
            # Apply effects based on component type
            if component.type == 'building':
                if outage_type == 'blackout':
                    # Complete power loss
                    component.state['energy_usage_kw'] = 0
                    component.state['hvac_active'] = False
                    
                    # Temperature will gradually approach outside temperature
                    outside_temp = 20  # Assumed outside temperature
                    component.state['temperature'] += (outside_temp - component.state['temperature']) * 0.1
                else:  # brownout
                    # Reduced power
                    component.state['energy_usage_kw'] *= 0.6  # 40% reduction
                    
                    # HVAC might still work but less effectively
                    if component.state['hvac_active']:
                        # Temperature control less effective
                        desired_temp = 22
                        component.state['temperature'] += (desired_temp - component.state['temperature']) * 0.05
            
            elif component.type == 'traffic_signal':
                if outage_type == 'blackout':
                    # Traffic signals not working
                    component.state['is_active'] = False
                    
                    # For intersections with no power, all connected roads get increased congestion
                    for road_id in component.connections:
                        if road_id in components and components[road_id].type == 'road':
                            road = components[road_id]
                            road.state['congestion'] = min(1.0, road.state['congestion'] + 0.3)
                            road.state['avg_speed'] *= 0.7  # 30% speed reduction
                            road.state['travel_time'] = road.properties['length'] / (road.state['avg_speed'] / 3.6)
                
                else:  # brownout
                    # Traffic signals might flash yellow
                    component.state['current_phase'] = 1  # Yellow/caution phase
                    component.state['is_active'] = True
                    
                    # Less severe impact on connected roads
                    for road_id in component.connections:
                        if road_id in components and components[road_id].type == 'road':
                            road = components[road_id]
                            road.state['congestion'] = min(1.0, road.state['congestion'] + 0.15)
                            road.state['avg_speed'] *= 0.85  # 15% speed reduction
                            road.state['travel_time'] = road.properties['length'] / (road.state['avg_speed'] / 3.6)
    
    def _apply_special_event_effects(self, components: Dict[str, CityComponent], time: float) -> None:
        """Apply effects of a special event (sports game, parade, etc.)."""
        event_type = self.properties.get('event_type', 'sports')
        location_id = self.properties.get('location_id')
        attendance = self.properties.get('attendance', 1000)
        
        # Find the event location
        location = None
        if location_id and location_id in components:
            location = components[location_id]
        
        # If no specific location, return
        if not location:
            return
        
        # Calculate the impact radius based on attendance
        radius = min(2000, max(500, attendance / 10))  # 500m - 2km radius
        
        # Apply effects to components near the event location
        for component_id, component in components.items():
            # Skip the event location itself
            if component_id == location_id:
                continue
                
            # Check if component is within radius
            if component.location and location.location:
                # Calculate distance
                dist = self._calculate_distance(
                    component.location.get('lat', 0),
                    component.location.get('lon', 0),
                    location.location.get('lat', 0),
                    location.location.get('lon', 0)
                )
                
                # Skip if too far
                if dist > radius:
                    continue
                
                # Scale effect by distance - closer means stronger effect
                distance_factor = 1.0 - (dist / radius)
                
                # Apply effects based on component type
                if component.type == 'road':
                    # Increased traffic near event
                    congestion_increase = 0.3 * distance_factor
                    component.state['congestion'] = min(1.0, component.state['congestion'] + congestion_increase)
                    
                    # Decreased speed
                    speed_factor = 1.0 - (0.4 * distance_factor)  # Up to 40% reduction
                    component.state['avg_speed'] *= speed_factor
                    component.state['travel_time'] = component.properties['length'] / (component.state['avg_speed'] / 3.6)
                    
                    # More vehicles
                    vehicle_increase = int(attendance * 0.1 * distance_factor)  # Assume 10% of attendance affects each road
                    component.state['vehicle_count'] += vehicle_increase
                
                elif component.type == 'building' and dist < radius * 0.5:
                    # Only nearby buildings are affected
                    if event_type == 'sports':
                        # Sports events often have nearby businesses open
                        if component.properties.get('type') == 'commercial':
                            # Increased occupancy in commercial buildings
                            occupancy_increase = int(component.properties['max_occupancy'] * 0.3 * distance_factor)
                            component.state['current_occupancy'] += occupancy_increase
                            
                            # More energy usage
                            component.state['energy_usage_kw'] *= (1.0 + 0.2 * distance_factor)
                    
                    elif event_type == 'parade':
                        # Parades affect all building types along the route
                        # Increased occupancy
                        occupancy_increase = int(component.properties['max_occupancy'] * 0.5 * distance_factor)
                        component.state['current_occupancy'] += occupancy_increase
                        
                        # More energy usage
                        component.state['energy_usage_kw'] *= (1.0 + 0.3 * distance_factor)
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate haversine distance between two points in meters.
        
        Args:
            lat1: Latitude of first point
            lon1: Longitude of first point
            lat2: Latitude of second point
            lon2: Longitude of second point
            
        Returns:
            Distance in meters
        """
        # Approximate radius of earth in meters
        R = 6371.0 * 1000
        
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)
        
        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad
        
        a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        
        distance = R * c
        return distance


class DigitalTwin:
    """
    Main class for the city digital twin simulation.
    
    This class manages the entire simulation, including components,
    scenarios, time progression, and data collection.
    """
    
    def __init__(self, city_name: str = "Default City", config: Dict = None):
        """
        Initialize the digital twin.
        
        Args:
            city_name: Name of the city
            config: Configuration dictionary
        """
        self.city_name = city_name
        self.config = config or {}
        
        # Get city bounds from config or use default
        self.bounds = self.config.get('bounds', DEFAULT_CITY_BOUNDS)
        
        # Components and scenarios
        self.components = {}
        self.scenarios = {}
        
        # Simulation state
        self.simulation_id = None
        self.simulation_time = 0
        self.simulation_start_time = None
        self.simulation_config = None
        self.is_running = False
        self.time_step = DEFAULT_TIME_STEP
        
        # Metrics and results
        self.metrics = {
            'traffic': {
                'avg_congestion': [],
                'avg_speed': [],
                'total_travel_time': []
            },
            'energy': {
                'total_usage': [],
                'building_usage': []
            },
            'emissions': {
                'co2': [],
                'nox': []
            }
        }
        
        # For parallel simulation
        self.simulation_thread = None
        self.stop_flag = threading.Event()
        self.simulation_results = queue.Queue()
        
        logger.info(f"Digital Twin initialized for {city_name}")
    
    def load_city_data(self, data_path: str) -> None:
        """
        Load city data from files.
        
        Args:
            data_path: Path to city data directory
        """
        try:
            # Check if directory exists
            if not os.path.isdir(data_path):
                logger.error(f"Data directory not found: {data_path}")
                return
                
            # Load components from JSON files
            components_path = os.path.join(data_path, "components")
            if os.path.isdir(components_path):
                for filename in os.listdir(components_path):
                    if filename.endswith(".json"):
                        file_path = os.path.join(components_path, filename)
                        with open(file_path, 'r') as f:
                            components_data = json.load(f)
                            
                            for comp_data in components_data:
                                self._create_component_from_data(comp_data)
            
            # Load scenarios
            scenarios_path = os.path.join(data_path, "scenarios")
            if os.path.isdir(scenarios_path):
                for filename in os.listdir(scenarios_path):
                    if filename.endswith(".json"):
                        file_path = os.path.join(scenarios_path, filename)
                        with open(file_path, 'r') as f:
                            scenarios_data = json.load(f)
                            
                            for scenario_data in scenarios_data:
                                self._create_scenario_from_data(scenario_data)
            
            # Load configuration
            config_path = os.path.join(data_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
                    
                    # Update bounds if provided
                    if 'bounds' in self.config:
                        self.bounds = self.config['bounds']
            
            logger.info(f"Loaded city data: {len(self.components)} components, {len(self.scenarios)} scenarios")
            
        except Exception as e:
            logger.error(f"Error loading city data: {str(e)}")
    
    def _create_component_from_data(self, data: Dict) -> None:
        """
        Create a component from data dictionary.
        
        Args:
            data: Component data dictionary
        """
        component_id = data.get('id')
        component_type = data.get('type')
        properties = data.get('properties', {})
        location = data.get('location')
        
        if not component_id or not component_type:
            logger.warning("Component missing ID or type, skipping")
            return
            
        # Create appropriate component type
        if component_type == 'traffic_signal':
            component = TrafficSignal(component_id, properties, location)
        elif component_type == 'building':
            component = Building(component_id, properties, location)
        elif component_type == 'road':
            component = Road(component_id, properties, location)
        else:
            # Generic component for other types
            component = CityComponent(component_id, component_type, properties, location)
            
        # Add connections if specified
        if 'connections' in data:
            for conn_id in data['connections']:
                component.connect_to(conn_id)
                
        # Add to components dictionary
        self.components[component_id] = component
    
    def _create_scenario_from_data(self, data: Dict) -> None:
        """
        Create a scenario from data dictionary.
        
        Args:
            data: Scenario data dictionary
        """
        scenario_id = data.get('id')
        scenario_type = data.get('type')
        properties = data.get('properties', {})
        
        if not scenario_id or not scenario_type:
            logger.warning("Scenario missing ID or type, skipping")
            return
            
        # Create scenario
        scenario = Scenario(scenario_id, scenario_type, properties)
        
        # Add to scenarios dictionary
        self.scenarios[scenario_id] = scenario
    
    def add_component(self, component: CityComponent) -> None:
        """
        Add a component to the digital twin.
        
        Args:
            component: Component to add
        """
        self.components[component.id] = component
        logger.info(f"Added component: {component.id} ({component.type})")
    
    def remove_component(self, component_id: str) -> bool:
        """
        Remove a component from the digital twin.
        
        Args:
            component_id: ID of component to remove
            
        Returns:
            True if component was removed, False otherwise
        """
        if component_id in self.components:
            # Remove from connections in other components
            for comp in self.components.values():
                if component_id in comp.connections:
                    comp.connections.remove(component_id)
            
            # Remove the component
            del self.components[component_id]
            logger.info(f"Removed component: {component_id}")
            return True
        else:
            logger.warning(f"Component not found: {component_id}")
            return False
    
    def add_scenario(self, scenario: Scenario) -> None:
        """
        Add a scenario to the digital twin.
        
        Args:
            scenario: Scenario to add
        """
        self.scenarios[scenario.id] = scenario
        logger.info(f"Added scenario: {scenario.id} ({scenario.type})")
    
    def remove_scenario(self, scenario_id: str) -> bool:
        """
        Remove a scenario from the digital twin.
        
        Args:
            scenario_id: ID of scenario to remove
            
        Returns:
            True if scenario was removed, False otherwise
        """
        if scenario_id in self.scenarios:
            del self.scenarios[scenario_id]
            logger.info(f"Removed scenario: {scenario_id}")
            return True
        else:
            logger.warning(f"Scenario not found: {scenario_id}")
            return False
    
    def start_simulation(self, config: Dict) -> str:
        """
        Start a new simulation.
        
        Args:
            config: Simulation configuration
            
        Returns:
            Simulation ID
        """
        if self.is_running:
            raise ValueError("A simulation is already running")
            
        # Generate simulation ID
        self.simulation_id = str(uuid.uuid4())
        
        # Parse and validate config
        self.simulation_config = config
        self.time_step = config.get('time_step', DEFAULT_TIME_STEP)
        
        # Set simulation start time
        start_time_str = config.get('start_time')
        if start_time_str:
            try:
                # Parse ISO format datetime
                import datetime
                self.simulation_start_time = datetime.datetime.fromisoformat(start_time_str)
            except ValueError:
                logger.warning(f"Invalid start_time format: {start_time_str}, using current time")
                self.simulation_start_time = datetime.datetime.now()
        else:
            # Use current time if not specified
            import datetime
            self.simulation_start_time = datetime.datetime.now()
        
        # Reset simulation time and metrics
        self.simulation_time = 0
        self.metrics = {
            'traffic': {
                'avg_congestion': [],
                'avg_speed': [],
                'total_travel_time': []
            },
            'energy': {
                'total_usage': [],
                'building_usage': []
            },
            'emissions': {
                'co2': [],
                'nox': []
            }
        }
        
        # Configure scenarios
        for scenario_config in config.get('scenarios', []):
            scenario_id = scenario_config.get('id')
            if scenario_id in self.scenarios:
                scenario = self.scenarios[scenario_id]
                
                # Set activation time
                activate_at = scenario_config.get('activate_at', 0)
                deactivate_at = scenario_config.get('deactivate_at')
                
                # Update scenario properties if provided
                if 'properties' in scenario_config:
                    scenario.properties.update(scenario_config['properties'])
                
                # Schedule activation/deactivation
                if activate_at >= 0:
                    scenario.activate(activate_at)
                    logger.info(f"Scheduled scenario {scenario_id} activation at time {activate_at}")
                
                if deactivate_at and deactivate_at > activate_at:
                    # Will be deactivated during simulation
                    logger.info(f"Scheduled scenario {scenario_id} deactivation at time {deactivate_at}")
            else:
                logger.warning(f"Scenario not found: {scenario_id}")
        
        # Start simulation in background thread
        self.stop_flag.clear()
        self.is_running = True
        self.simulation_thread = threading.Thread(target=self._run_simulation)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
        
        logger.info(f"Started simulation {self.simulation_id}")
        return self.simulation_id
    
    def stop_simulation(self) -> None:
        """Stop the current simulation."""
        if not self.is_running:
            logger.warning("No simulation is running")
            return
            
        # Set stop flag to terminate the simulation thread
        self.stop_flag.set()
        
        # Wait for thread to finish (with timeout)
        if self.simulation_thread:
            self.simulation_thread.join(timeout=5.0)
            
        self.is_running = False
        logger.info(f"Stopped simulation {self.simulation_id}")
    
    def _run_simulation(self) -> None:
        """Run the simulation in a background thread."""
        try:
            # Get simulation parameters
            duration = self.simulation_config.get('duration', 3600)  # Default 1 hour
            duration = min(duration, MAX_SIMULATION_DURATION)  # Limit to maximum
            
            # Time step for simulation
            time_step = self.time_step
            
            # Get random seed if specified
            random_seed = self.simulation_config.get('random_seed')
            if random_seed is not None:
                np.random.seed(random_seed)
            
            # Main simulation loop
            while self.simulation_time < duration and not self.stop_flag.is_set():
                # Update all components
                self._update_components(time_step)
                
                # Apply active scenarios
                self._apply_scenarios()
                
                # Calculate and store metrics
                self._calculate_metrics()
                
                # Update simulation time
                self.simulation_time += time_step
                
                # Emit progress update
                progress = min(100.0, (self.simulation_time / duration) * 100)
                self._emit_progress(progress)
                
                # Throttle simulation speed if needed
                # time.sleep(0.001)  # Small sleep to prevent CPU overload
            
            # Simulation complete
            if not self.stop_flag.is_set():
                # Normal completion (not stopped)
                self._emit_completion()
            
        except Exception as e:
            logger.error(f"Error in simulation: {str(e)}")
            self._emit_error(str(e))
            
        finally:
            self.is_running = False
    
    def _update_components(self, time_step: float) -> None:
        """
        Update all components for one simulation step.
        
        Args:
            time_step: Simulation time step in seconds
        """
        # Get current hour for time-based patterns
        import datetime
        sim_datetime = self.simulation_start_time + datetime.timedelta(seconds=self.simulation_time)
        hour = sim_datetime.hour
        day_type = 'weekend' if sim_datetime.weekday() >= 5 else 'weekday'
        
        # Get traffic scale factor
        traffic_scale = self.simulation_config.get('traffic_scale', 1.0)
        
        # Get outside temperature (could be based on weather data)
        temperature = 20 + 5 * np.sin(np.pi * (hour - 6) / 12)  # Simple model: 15-25°C daily cycle
        
        # Create environment state dictionary
        environment_state = {
            'hour': hour,
            'day_type': day_type,
            'traffic_scale': traffic_scale,
            'temperature': temperature,
            'connected_roads': {}
        }
        
        # First pass: get road states for connectivity
        for component_id, component in self.components.items():
            if component.type == 'road':
                environment_state['connected_roads'][component_id] = component.state
        
        # Second pass: update all components
        for component_id, component in self.components.items():
            component.update(time_step, environment_state)
            component.save_state(self.simulation_time)
    
    def _apply_scenarios(self) -> None:
        """Apply active scenarios to the simulation."""
        # Activate/deactivate scenarios based on time
        for scenario_id, scenario in self.scenarios.items():
            # Check for pending activation
            if not scenario.active and scenario.start_time is not None:
                if self.simulation_time >= scenario.start_time:
                    logger.info(f"Activating scenario {scenario_id} at time {self.simulation_time}")
                    scenario.activate(self.simulation_time)
            
            # Check for pending deactivation
            elif scenario.active and scenario.properties.get('duration'):
                duration = scenario.properties['duration']
                if self.simulation_time >= scenario.start_time + duration:
                    logger.info(f"Deactivating scenario {scenario_id} at time {self.simulation_time}")
                    scenario.deactivate(self.simulation_time)
            
            # Apply active scenarios
            if scenario.active:
                scenario.apply_effects(self.components, self.simulation_time)
    
    def _calculate_metrics(self) -> None:
        """Calculate and store simulation metrics."""
        # Traffic metrics
        road_components = [c for c in self.components.values() if c.type == 'road']
        building_components = [c for c in self.components.values() if c.type == 'building']
        
        if road_components:
            avg_congestion = np.mean([r.state['congestion'] for r in road_components])
            avg_speed = np.mean([r.state['avg_speed'] for r in road_components])
            total_travel_time = np.sum([r.state['travel_time'] for r in road_components])
            
            self.metrics['traffic']['avg_congestion'].append((self.simulation_time, avg_congestion))
            self.metrics['traffic']['avg_speed'].append((self.simulation_time, avg_speed))
            self.metrics['traffic']['total_travel_time'].append((self.simulation_time, total_travel_time))
        
        # Energy metrics
        if building_components:
            total_energy = np.sum([b.state['energy_usage_kw'] for b in building_components])
            building_energy = {}
            for b in building_components:
                building_energy[b.id] = b.state['energy_usage_kw']
                
            self.metrics['energy']['total_usage'].append((self.simulation_time, total_energy))
            self.metrics['energy']['building_usage'].append((self.simulation_time, building_energy))
        
        # Emissions metrics (simplified calculation)
        if road_components:
            # Simplified CO2 calculation based on traffic
            co2 = 0
            for road in road_components:
                # Higher congestion = higher emissions
                congestion_factor = 1.0 + road.state['congestion'] * 2.0
                vehicle_count = road.state['vehicle_count']
                
                # Approximate emissions (kg CO2)
                road_co2 = vehicle_count * 0.2 * congestion_factor * (self.time_step / 3600)
                co2 += road_co2
            
            # NOx is roughly proportional to CO2 but affected more by congestion
            nox = co2 * 0.001 * (1.0 + avg_congestion)
            
            self.metrics['emissions']['co2'].append((self.simulation_time, co2))
            self.metrics['emissions']['nox'].append((self.simulation_time, nox))
    
    def _emit_progress(self, progress: float) -> None:
        """
        Emit progress update.
        
        Args:
            progress: Progress percentage (0-100)
        """
        status = {
            'id': self.simulation_id,
            'status': 'running',
            'progress': progress,
            'current_time': self.simulation_time,
            'message': f"Simulation in progress: {progress:.1f}%"
        }
        self.simulation_results.put(status)
    
    def _emit_completion(self) -> None:
        """Emit simulation completion."""
        final_metrics = self._summarize_metrics()
        
        status = {
            'id': self.simulation_id,
            'status': 'completed',
            'progress': 100.0,
            'current_time': self.simulation_time,
            'message': "Simulation completed successfully",
            'metrics': final_metrics
        }
        self.simulation_results.put(status)
    
    def _emit_error(self, error_message: str) -> None:
        """
        Emit simulation error.
        
        Args:
            error_message: Error message
        """
        status = {
            'id': self.simulation_id,
            'status': 'failed',
            'progress': 0.0,
            'message': f"Simulation failed: {error_message}"
        }
        self.simulation_results.put(status)
    
    def _summarize_metrics(self) -> Dict:
        """
        Summarize metrics from the simulation.
        
        Returns:
            Dictionary with summarized metrics
        """
        summary = {}
        
        # Traffic metrics
        if self.metrics['traffic']['avg_congestion']:
            congestion_values = [v for _, v in self.metrics['traffic']['avg_congestion']]
            speed_values = [v for _, v in self.metrics['traffic']['avg_speed']]
            
            summary['traffic'] = {
                'avg_congestion': np.mean(congestion_values),
                'max_congestion': np.max(congestion_values),
                'min_congestion': np.min(congestion_values),
                'avg_speed': np.mean(speed_values),
                'time_series': {
                    'congestion': self.metrics['traffic']['avg_congestion'][-20:],  # Last 20 points
                    'speed': self.metrics['traffic']['avg_speed'][-20:]
                }
            }
        
        # Energy metrics
        if self.metrics['energy']['total_usage']:
            energy_values = [v for _, v in self.metrics['energy']['total_usage']]
            
            summary['energy'] = {
                'avg_usage_kw': np.mean(energy_values),
                'max_usage_kw': np.max(energy_values),
                'total_kwh': np.sum(energy_values) * (self.time_step / 3600),
                'time_series': {
                    'usage': self.metrics['energy']['total_usage'][-20:]
                }
            }
        
        # Emissions metrics
        if self.metrics['emissions']['co2']:
            co2_values = [v for _, v in self.metrics['emissions']['co2']]
            nox_values = [v for _, v in self.metrics['emissions']['nox']]
            
            summary['emissions'] = {
                'total_co2_kg': np.sum(co2_values),
                'total_nox_kg': np.sum(nox_values),
                'time_series': {
                    'co2': self.metrics['emissions']['co2'][-20:],
                    'nox': self.metrics['emissions']['nox'][-20:]
                }
            }
        
        # Add scenario impact metrics
        active_scenarios = [s for s in self.scenarios.values() if s.active]
        if active_scenarios:
            scenario_impacts = {}
            
            for scenario in active_scenarios:
                # Calculate approximate impact of this scenario
                impact = {
                    'type': scenario.type,
                    'duration': self.simulation_time - scenario.start_time
                }
                
                # Specific impacts based on scenario type
                if scenario.type == 'traffic_accident':
                    road_id = scenario.properties.get('road_id')
                    if road_id in self.components:
                        road = self.components[road_id]
                        impact['congestion_increase'] = road.state['congestion']
                        impact['speed_reduction'] = road.properties['max_speed'] - road.state['avg_speed']
                
                elif scenario.type == 'weather_event':
                    impact['energy_effect'] = summary['energy']['avg_usage_kw'] if 'energy' in summary else 0
                    
                elif scenario.type == 'power_outage':
                    impact['affected_buildings'] = len([c for c in self.components.values() 
                                                     if c.type == 'building' and c.state.get('energy_usage_kw', 0) == 0])
                
                scenario_impacts[scenario.id] = impact
            
            summary['scenario_impacts'] = scenario_impacts
        
        return summary
    
    def get_simulation_status(self) -> Dict:
        """
        Get the current simulation status.
        
        Returns:
            Status dictionary
        """
        if not self.simulation_id:
            return {
                'id': None,
                'status': 'idle',
                'progress': 0.0,
                'message': "No simulation running"
            }
        
        # Check if there's a status update in the queue
        try:
            status = self.simulation_results.get_nowait()
            return status
        except queue.Empty:
            # No update, return current status
            progress = 0.0
            if self.simulation_config and 'duration' in self.simulation_config:
                progress = min(100.0, (self.simulation_time / self.simulation_config['duration']) * 100)
                
            import datetime
            sim_datetime = self.simulation_start_time + datetime.timedelta(seconds=self.simulation_time)
                
            return {
                'id': self.simulation_id,
                'status': 'running' if self.is_running else 'completed',
                'progress': progress,
                'current_time': self.simulation_time,
                'sim_datetime': sim_datetime.isoformat(),
                'message': f"Simulation in progress: {progress:.1f}%"
            }
    
    def get_component_state(self, component_id: str, time: float = None) -> Dict:
        """
        Get the state of a component.
        
        Args:
            component_id: ID of the component
            time: Optional time to get state for
            
        Returns:
            Component state dictionary
        """
        if component_id not in self.components:
            raise ValueError(f"Component not found: {component_id}")
            
        component = self.components[component_id]
        
        if time is not None:
            # Get historical state
            state = component.get_state_at(time)
        else:
            # Get current state
            state = component.state
            
        return {
            'id': component_id,
            'type': component.type,
            'state': state,
            'timestamp': time if time is not None else self.simulation_time
        }
    
    def query_simulation_data(self, query: Dict) -> Dict:
        """
        Query simulation data based on parameters.
        
        Args:
            query: Query parameters
            
        Returns:
            Query results
        """
        # Extract query parameters
        component_types = query.get('component_types')
        component_ids = query.get('component_ids')
        time_range = query.get('time_range')
        metrics_list = query.get('metrics')
        aggregation = query.get('aggregation')
        
        # Filter components
        filtered_components = {}
        
        for component_id, component in self.components.items():
            # Filter by type
            if component_types and component.type not in component_types:
                continue
                
            # Filter by ID
            if component_ids and component_id not in component_ids:
                continue
                
            # Add to filtered list
            filtered_components[component_id] = component
        
        # Extract data based on parameters
        results = {
            'components': {},
            'metrics': {}
        }
        
        # Component state data
        for component_id, component in filtered_components.items():
            # Get complete state history or subset based on time range
            if time_range:
                start_time, end_time = time_range
                history = [entry for entry in component.state_history 
                         if start_time <= entry['timestamp'] <= end_time]
            else:
                history = component.state_history
            
            # Add to results
            results['components'][component_id] = {
                'type': component.type,
                'state_history': history
            }
        
        # Metrics data
        if metrics_list:
            for metric_name in metrics_list:
                # Parse metric path (e.g., 'traffic.avg_congestion')
                parts = metric_name.split('.')
                if len(parts) == 2 and parts[0] in self.metrics and parts[1] in self.metrics[parts[0]]:
                    metric_data = self.metrics[parts[0]][parts[1]]
                    
                    # Filter by time range if specified
                    if time_range:
                        start_time, end_time = time_range
                        metric_data = [(t, v) for t, v in metric_data if start_time <= t <= end_time]
                    
                    # Apply aggregation if requested
                    if aggregation:
                        if aggregation == 'avg':
                            value = np.mean([v for _, v in metric_data]) if metric_data else 0
                        elif aggregation == 'sum':
                            value = np.sum([v for _, v in metric_data]) if metric_data else 0
                        elif aggregation == 'max':
                            value = np.max([v for _, v in metric_data]) if metric_data else 0
                        elif aggregation == 'min':
                            value = np.min([v for _, v in metric_data]) if metric_data else 0
                        else:
                            value = metric_data
                            
                        results['metrics'][metric_name] = value
                    else:
                        results['metrics'][metric_name] = metric_data
        
        return results
    
    def save_simulation_results(self, output_path: str) -> None:
        """
        Save simulation results to a file.
        
        Args:
            output_path: Path to save results
        """
        # Create output directory if needed
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Collect results to save
        results = {
            'simulation_id': self.simulation_id,
            'city_name': self.city_name,
            'config': self.simulation_config,
            'time': self.simulation_time,
            'metrics': self._summarize_metrics(),
            'scenarios': {id: {'type': s.type, 'active': s.active} for id, s in self.scenarios.items()},
            'timestamp': time.time()
        }
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Simulation results saved to {output_path}")


# API setup
app = FastAPI(title="City Digital Twin API", 
             description="API for the Smart City Digital Twin simulation",
             version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create digital twin instance
digital_twin = DigitalTwin("Smart City")

# Routes
@app.get("/")
async def root():
    """API root endpoint with basic info."""
    return {
        "name": "City Digital Twin API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/status")
async def get_status():
    """Get the overall status of the Digital Twin."""
    component_count = len(digital_twin.components)
    scenario_count = len(digital_twin.scenarios)
    
    simulation_status = digital_twin.get_simulation_status()
    
    return {
        "city_name": digital_twin.city_name,
        "component_count": component_count,
        "scenario_count": scenario_count,
        "simulation": simulation_status
    }

@app.post("/simulations", response_model=Dict)
async def start_simulation(config: SimulationConfig):
    """Start a new simulation with the provided configuration."""
    try:
        simulation_id = digital_twin.start_simulation(config.dict())
        return {
            "simulation_id": simulation_id,
            "status": "started",
            "message": "Simulation started successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/simulations/{simulation_id}")
async def get_simulation(simulation_id: str):
    """Get the status of a specific simulation."""
    status = digital_twin.get_simulation_status()
    
    if status['id'] != simulation_id:
        raise HTTPException(status_code=404, detail="Simulation not found")
        
    return status

@app.delete("/simulations/{simulation_id}")
async def stop_simulation(simulation_id: str):
    """Stop a running simulation."""
    status = digital_twin.get_simulation_status()
    
    if status['id'] != simulation_id:
        raise HTTPException(status_code=404, detail="Simulation not found")
        
    if status['status'] != 'running':
        return {
            "simulation_id": simulation_id,
            "status": status['status'],
            "message": "Simulation is not running"
        }
        
    digital_twin.stop_simulation()
    
    return {
        "simulation_id": simulation_id,
        "status": "stopped",
        "message": "Simulation stopped successfully"
    }

@app.get("/components")
async def get_components(type: str = None):
    """Get all components or filter by type."""
    components = {}
    
    for component_id, component in digital_twin.components.items():
        if type is None or component.type == type:
            components[component_id] = component.to_dict()
            
    return {
        "component_count": len(components),
        "components": components
    }

@app.get("/components/{component_id}")
async def get_component(component_id: str):
    """Get a specific component by ID."""
    if component_id not in digital_twin.components:
        raise HTTPException(status_code=404, detail="Component not found")
        
    return digital_twin.components[component_id].to_dict()

@app.post("/components")
async def add_component(component: ComponentConfig):
    """Add a new component to the digital twin."""
    # Check if component already exists
    if component.component_id in digital_twin.components:
        raise HTTPException(status_code=400, detail="Component ID already exists")
        
    # Create the appropriate component type
    if component.component_type == 'traffic_signal':
        new_component = TrafficSignal(component.component_id, component.properties, component.location)
    elif component.component_type == 'building':
        new_component = Building(component.component_id, component.properties, component.location)
    elif component.component_type == 'road':
        new_component = Road(component.component_id, component.properties, component.location)
    else:
        new_component = CityComponent(component.component_id, component.component_type, 
                                    component.properties, component.location)
    
    # Add connections
    for connected_id in component.connections:
        new_component.connect_to(connected_id)
        
    # Add to digital twin
    digital_twin.add_component(new_component)
    
    return {
        "component_id": component.component_id,
        "status": "added",
        "message": f"Component added successfully"
    }

@app.delete("/components/{component_id}")
async def remove_component(component_id: str):
    """Remove a component from the digital twin."""
    result = digital_twin.remove_component(component_id)
    
    if not result:
        raise HTTPException(status_code=404, detail="Component not found")
        
    return {
        "component_id": component_id,
        "status": "removed",
        "message": "Component removed successfully"
    }

@app.get("/scenarios")
async def get_scenarios(type: str = None):
    """Get all scenarios or filter by type."""
    scenarios = {}
    
    for scenario_id, scenario in digital_twin.scenarios.items():
        if type is None or scenario.type == type:
            scenarios[scenario_id] = {
                "id": scenario.id,
                "type": scenario.type,
                "properties": scenario.properties,
                "active": scenario.active
            }
            
    return {
        "scenario_count": len(scenarios),
        "scenarios": scenarios
    }

@app.post("/query")
async def query_data(query: QueryParams):
    """Query simulation data based on parameters."""
    try:
        results = digital_twin.query_simulation_data(query.dict())
        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/simulations/{simulation_id}/save")
async def save_results(simulation_id: str, output_path: str = Body(..., embed=True)):
    """Save simulation results to a file."""
    status = digital_twin.get_simulation_status()
    
    if status['id'] != simulation_id:
        raise HTTPException(status_code=404, detail="Simulation not found")
        
    try:
        digital_twin.save_simulation_results(output_path)
        return {
            "simulation_id": simulation_id,
            "output_path": output_path,
            "status": "saved",
            "message": "Simulation results saved successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/load-city")
async def load_city_data(data_path: str = Body(..., embed=True)):
    """Load city data from a directory."""
    try:
        digital_twin.load_city_data(data_path)
        return {
            "status": "loaded",
            "component_count": len(digital_twin.components),
            "scenario_count": len(digital_twin.scenarios),
            "message": "City data loaded successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the API server."""
    uvicorn.run("digital_twin:app", host=host, port=port, reload=True)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='City Digital Twin API')
    parser.add_argument('--host', type=str, default="0.0.0.0", help='Host to run server on')
    parser.add_argument('--port', type=int, default=8000, help='Port to run server on')
    parser.add_argument('--data', type=str, help='Path to city data directory')
    
    args = parser.parse_args()
    
    # Load city data if provided
    if args.data:
        digital_twin.load_city_data(args.data)
    
    # Run server
    run_server(args.host, args.port)

"""
SUMMARY:
========
This module implements a City Digital Twin API that creates a virtual model of a city 
for testing AI optimizations before real-world deployment. The system simulates various 
city components including traffic signals, roads, buildings, and scenarios like accidents 
or weather events.

Key components:
1. CityComponent - Base class for all city infrastructure elements
2. Scenario - Events that affect normal city operations
3. DigitalTwin - Core simulation engine that manages components and time progression
4. FastAPI endpoints - RESTful API for interacting with the digital twin

The API allows other smart city systems to test their algorithms and optimizations
in a realistic simulated environment before deploying them in the real world.

TODO:
=====
1. Implement more sophisticated traffic flow models using cellular automata
2. Add pedestrian simulation with realistic crowd dynamics
3. Create integration points for external AI optimization systems
4. Add geospatial visualization capabilities using Mapbox or Leaflet
5. Implement more detailed building energy models with HVAC simulation
6. Add support for electrical grid simulation with load balancing
7. Create a simple UI for scenario creation and simulation monitoring
8. Improve simulation performance with parallel component updates
9. Add support for importing real city data from OpenStreetMap
10. Implement machine learning models to learn from real city data
11. Create a WebSocket API for real-time simulation updates
12. Add support for time-series database integration (InfluxDB/TimescaleDB)
13. Implement validation against real-world sensor data
"""
