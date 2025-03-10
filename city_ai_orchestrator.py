"""
The brain of our smart city - coordinates traffic optimization, waste collection,
energy distribution, and emergency response systems to ensure they work in harmony
rather than fighting each other for resources.

I've built this as a central coordinator to handle the inevitable conflicts that
arise when multiple AI systems try to optimize their own domains independently.
"""

import asyncio
import datetime
import json
import logging
import os
import pickle
import random
import signal
import sys
import threading
import time
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Any, Optional, Set, Union

import aiohttp
import networkx as nx
import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, BackgroundTasks, Depends, HTTPException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("orchestrator.log"), logging.StreamHandler()]
)
logger = logging.getLogger("CityOrchestrator")

# Constants
OPTIMIZATION_INTERVAL = 15 * 60  # seconds (15 minutes)
STATUS_UPDATE_INTERVAL = 5 * 60  # seconds (5 minutes)
MAX_EXECUTION_TIME = 5 * 60  # seconds (5 minutes)

# Priority and weight constants
PRIORITY_LEVELS = {
    "emergency": 100,
    "high": 75, 
    "medium": 50,
    "low": 25
}

# Default optimization weights (must sum to 1.0)
DEFAULT_WEIGHTS = {
    "traffic": 0.30,
    "energy": 0.30,
    "waste": 0.15,
    "emergency": 0.25
}


class Subsystem:
    """Base class for all smart city subsystems handled by the orchestrator."""
    
    def __init__(self, system_id: str, system_type: str, config: dict = None):
        """
        Initialize a subsystem.
        
        Args:
            system_id: Unique identifier for this subsystem
            system_type: Type of subsystem ('traffic', 'energy', etc.)
            config: Configuration dictionary
        """
        self.id = system_id
        self.type = system_type
        self.config = config or {}
        self.status = "idle"
        self.last_update = datetime.datetime.now()
        self.current_metrics = {}
        self.optimization_history = []
        self.api_endpoint = self.config.get("api_endpoint")
        self.api_key = self.config.get("api_key")
        
        # Dependencies on other subsystems
        self.dependencies = []
        
        logger.debug(f"Initialized {system_type} subsystem: {system_id}")
    
    async def get_status(self) -> dict:
        """
        Get the current status of the subsystem.
        
        Returns:
            Dictionary with status information
        """
        # Try to get status from API if available
        if self.api_endpoint:
            try:
                async with aiohttp.ClientSession() as session:
                    headers = {}
                    if self.api_key:
                        headers["Authorization"] = f"Bearer {self.api_key}"
                    
                    async with session.get(f"{self.api_endpoint}/status", headers=headers) as response:
                        if response.status == 200:
                            data = await response.json()
                            self.current_metrics = data.get("metrics", {})
                            return {
                                "id": self.id,
                                "type": self.type,
                                "status": data.get("status", self.status),
                                "metrics": self.current_metrics,
                                "last_update": datetime.datetime.now().isoformat()
                            }
            except Exception as e:
                logger.error(f"Error getting status for {self.id}: {e}")
        
        # Return cached data if API call fails or no API
        return {
            "id": self.id,
            "type": self.type,
            "status": self.status,
            "metrics": self.current_metrics,
            "last_update": self.last_update.isoformat()
        }
    
    async def optimize(self, constraints: dict = None) -> dict:
        """
        Run optimization for this subsystem.
        
        Args:
            constraints: Dictionary of constraints from the orchestrator
            
        Returns:
            Optimization results
        """
        # To be implemented by subclasses
        # This is where each subsystem would run its specific optimization algorithm
        
        logger.warning(f"Base optimize method called for {self.id}, should be overridden")
        return {
            "status": "error",
            "message": "Not implemented in base class",
            "metrics": {}
        }
    
    async def apply_optimization(self, plan: dict) -> dict:
        """
        Apply an optimization plan to the subsystem.
        
        Args:
            plan: Optimization plan to apply
            
        Returns:
            Application results
        """
        # Try to apply via API if available
        if self.api_endpoint:
            try:
                async with aiohttp.ClientSession() as session:
                    headers = {
                        "Content-Type": "application/json"
                    }
                    if self.api_key:
                        headers["Authorization"] = f"Bearer {self.api_key}"
                    
                    async with session.post(
                        f"{self.api_endpoint}/apply",
                        json=plan,
                        headers=headers
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            return {
                                "status": "success",
                                "message": "Applied optimization plan",
                                "details": result
                            }
                        else:
                            return {
                                "status": "error",
                                "message": f"API error: {response.status}",
                                "details": await response.text()
                            }
            except Exception as e:
                logger.error(f"Error applying optimization to {self.id}: {e}")
                return {
                    "status": "error",
                    "message": str(e),
                    "details": None
                }
        
        # If no API, just return success (simulation)
        logger.info(f"Simulated applying optimization plan to {self.id}")
        return {
            "status": "success",
            "message": "Simulated optimization application",
            "details": None
        }
    
    def add_dependency(self, subsystem_id: str) -> None:
        """
        Add a dependency on another subsystem.
        
        Args:
            subsystem_id: ID of the subsystem this one depends on
        """
        if subsystem_id not in self.dependencies:
            self.dependencies.append(subsystem_id)
    
    def to_dict(self) -> dict:
        """
        Convert to dictionary representation.
        
        Returns:
            Dictionary with subsystem data
        """
        return {
            "id": self.id,
            "type": self.type,
            "status": self.status,
            "last_update": self.last_update.isoformat(),
            "current_metrics": self.current_metrics,
            "dependencies": self.dependencies,
            # Don't include API key in output
            "config": {k: v for k, v in self.config.items() if k != "api_key"}
        }


class TrafficSubsystem(Subsystem):
    """Traffic management subsystem - controls traffic signals and routing."""
    
    def __init__(self, system_id: str, config: dict = None):
        super().__init__(system_id, "traffic", config)
        
        # Traffic-specific attributes
        self.control_algorithm = self.config.get("control_algorithm", "reinforcement_learning")
        self.signal_count = self.config.get("signal_count", 50)
        self.congestion_threshold = self.config.get("congestion_threshold", 0.7)
        
        # Default metrics
        self.current_metrics = {
            "average_wait_time": 45.0,  # seconds
            "average_speed": 32.0,  # km/h
            "congestion_level": 0.45,  # 0-1 scale
            "emissions": 320.0,  # kg CO2 equivalent
            "fuel_consumption": 580.0,  # liters
            "throughput": 1200  # vehicles per hour
        }
    
    async def optimize(self, constraints: dict = None) -> dict:
        """Run traffic optimization with given constraints."""
        start_time = time.time()
        constraints = constraints or {}
        
        try:
            # Use the API if available
            if self.api_endpoint:
                # Implementation for real API call would go here
                pass
            
            # Simulated optimization for demonstration
            # In a real implementation, this would run a traffic optimization algorithm
            
            # Apply scenario constraints
            emergency_mode = constraints.get("emergency_mode", False)
            restricted_areas = constraints.get("restricted_areas", [])
            energy_saving = constraints.get("energy_saving", False)
            
            # Generate improvement metrics
            # In a real implementation, these would come from the optimization algorithm
            improvements = {
                "wait_time_reduction": 12.5 if not emergency_mode else 25.0,
                "speed_increase": 4.0 if not emergency_mode else 8.0,
                "congestion_reduction": 0.15 if not emergency_mode else 0.25,
                "emissions_reduction": 35.0 if energy_saving else 15.0
            }
            
            # Update metrics
            new_metrics = {
                "average_wait_time": max(15.0, self.current_metrics["average_wait_time"] - improvements["wait_time_reduction"]),
                "average_speed": min(60.0, self.current_metrics["average_speed"] + improvements["speed_increase"]),
                "congestion_level": max(0.1, self.current_metrics["congestion_level"] - improvements["congestion_reduction"]),
                "emissions": max(100.0, self.current_metrics["emissions"] - improvements["emissions_reduction"]),
                "fuel_consumption": max(200.0, self.current_metrics["fuel_consumption"] * 0.9),
                "throughput": self.current_metrics["throughput"] * 1.2
            }
            
            self.current_metrics = new_metrics
            self.last_update = datetime.datetime.now()
            
            # Generate optimization plan
            plan = {
                "algorithm": self.control_algorithm,
                "signal_timings": {
                    "green_wave": not emergency_mode,
                    "emergency_priority": emergency_mode
                },
                "routing": {
                    "avoid_areas": restricted_areas,
                    "energy_efficient": energy_saving
                }
            }
            
            # Add to optimization history
            self.optimization_history.append({
                "timestamp": datetime.datetime.now().isoformat(),
                "constraints": constraints,
                "improvements": improvements,
                "metrics": new_metrics
            })
            
            # Keep history manageable
            if len(self.optimization_history) > 100:
                self.optimization_history = self.optimization_history[-100:]
            
            return {
                "status": "success",
                "message": "Traffic optimization completed",
                "metrics": self.current_metrics,
                "improvements": improvements,
                "plan": plan,
                "execution_time": time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"Error in traffic optimization: {e}")
            return {
                "status": "error",
                "message": str(e),
                "metrics": self.current_metrics,
                "execution_time": time.time() - start_time
            }


class EnergySubsystem(Subsystem):
    """Energy management subsystem - controls power distribution and usage."""
    
    def __init__(self, system_id: str, config: dict = None):
        super().__init__(system_id, "energy", config)
        
        # Energy-specific attributes
        self.grid_model = self.config.get("grid_model", "smart_grid")
        self.has_renewables = self.config.get("has_renewables", True)
        self.battery_storage = self.config.get("battery_storage", False)
        
        # Default metrics
        self.current_metrics = {
            "total_consumption": 120.0,  # MWh
            "renewable_percentage": 22.0,  # percent
            "grid_stability": 0.92,  # 0-1 scale
            "peak_demand": 180.0,  # MW
            "cost_per_kwh": 0.14,  # currency units
            "carbon_intensity": 420.0  # g CO2/kWh
        }
    
    async def optimize(self, constraints: dict = None) -> dict:
        """Run energy optimization with given constraints."""
        start_time = time.time()
        constraints = constraints or {}
        
        try:
            # Use the API if available
            if self.api_endpoint:
                # Implementation for real API call would go here
                pass
            
            # Simulated optimization for demonstration
            # In a real implementation, this would run an energy optimization algorithm
            
            # Apply scenario constraints
            critical_services = constraints.get("critical_services", [])
            max_consumption = constraints.get("max_consumption")
            storm_protection = constraints.get("storm_protection", False)
            
            # Generate improvement metrics
            # In a real implementation, these would come from the optimization algorithm
            improvements = {
                "renewable_increase": 3.5 if self.has_renewables else 0.0,
                "consumption_reduction": 8.0 if max_consumption else 3.0,
                "stability_change": -0.03 if storm_protection else 0.01,
                "peak_reduction": 12.0 if max_consumption else 5.0
            }
            
            # Update metrics
            new_metrics = {
                "total_consumption": max(80.0, self.current_metrics["total_consumption"] - improvements["consumption_reduction"]),
                "renewable_percentage": min(90.0, self.current_metrics["renewable_percentage"] + improvements["renewable_increase"]),
                "grid_stability": max(0.8, min(0.99, self.current_metrics["grid_stability"] + improvements["stability_change"])),
                "peak_demand": max(120.0, self.current_metrics["peak_demand"] - improvements["peak_reduction"]),
                "cost_per_kwh": max(0.09, self.current_metrics["cost_per_kwh"] * 0.98),
                "carbon_intensity": max(100.0, self.current_metrics["carbon_intensity"] * (1.0 - improvements["renewable_increase"]/100))
            }
            
            self.current_metrics = new_metrics
            self.last_update = datetime.datetime.now()
            
            # Generate optimization plan
            plan = {
                "grid_model": self.grid_model,
                "load_balancing": {
                    "critical_services": critical_services,
                    "demand_response": max_consumption is not None
                },
                "generation": {
                    "target_renewable": self.current_metrics["renewable_percentage"],
                    "storage_discharge": self.battery_storage and storm_protection
                },
                "protection": {
                    "storm_mode": storm_protection
                }
            }
            
            # Add to optimization history
            self.optimization_history.append({
                "timestamp": datetime.datetime.now().isoformat(),
                "constraints": constraints,
                "improvements": improvements,
                "metrics": new_metrics
            })
            
            # Keep history manageable
            if len(self.optimization_history) > 100:
                self.optimization_history = self.optimization_history[-100:]
            
            return {
                "status": "success",
                "message": "Energy optimization completed",
                "metrics": self.current_metrics,
                "improvements": improvements,
                "plan": plan,
                "execution_time": time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"Error in energy optimization: {e}")
            return {
                "status": "error",
                "message": str(e),
                "metrics": self.current_metrics,
                "execution_time": time.time() - start_time
            }


class WasteSubsystem(Subsystem):
    """Waste management subsystem - controls collection routes and scheduling."""
    
    def __init__(self, system_id: str, config: dict = None):
        super().__init__(system_id, "waste", config)
        
        # Waste-specific attributes
        self.routing_algorithm = self.config.get("routing_algorithm", "dynamic")
        self.has_sensors = self.config.get("has_sensors", True)
        self.vehicle_count = self.config.get("vehicle_count", 15)
        
        # Default metrics
        self.current_metrics = {
            "collection_efficiency": 0.72,  # 0-1 scale
            "fuel_usage": 450.0,  # liters
            "average_fill_level": 0.68,  # 0-1 scale
            "bins_collected": 830,  # count
            "distance_traveled": 320.0,  # km
            "overflow_incidents": 5  # count
        }
    
    async def optimize(self, constraints: dict = None) -> dict:
        """Run waste collection optimization with given constraints."""
        start_time = time.time()
        constraints = constraints or {}
        
        try:
            # Use the API if available
            if self.api_endpoint:
                # Implementation for real API call would go here
                pass
            
            # Simulated optimization for demonstration
            # In a real implementation, this would run a waste route optimization algorithm
            
            # Apply scenario constraints
            avoid_areas = constraints.get("avoid_areas", [])
            max_fuel = constraints.get("max_fuel")
            high_priority_areas = constraints.get("high_priority_areas", [])
            
            # Generate improvement metrics
            # In a real implementation, these would come from the optimization algorithm
            improvements = {
                "efficiency_increase": 0.08 if self.has_sensors else 0.04,
                "fuel_reduction": 40.0 if max_fuel else 20.0,
                "fill_level_reduction": 0.12 if self.has_sensors else 0.06,
                "overflow_reduction": 2 if high_priority_areas else 1
            }
            
            # Update metrics
            new_metrics = {
                "collection_efficiency": min(0.95, self.current_metrics["collection_efficiency"] + improvements["efficiency_increase"]),
                "fuel_usage": max(300.0, self.current_metrics["fuel_usage"] - improvements["fuel_reduction"]),
                "average_fill_level": max(0.3, self.current_metrics["average_fill_level"] - improvements["fill_level_reduction"]),
                "bins_collected": self.current_metrics["bins_collected"] * 1.05,
                "distance_traveled": self.current_metrics["distance_traveled"] * 0.9,
                "overflow_incidents": max(0, self.current_metrics["overflow_incidents"] - improvements["overflow_reduction"])
            }
            
            self.current_metrics = new_metrics
            self.last_update = datetime.datetime.now()
            
            # Generate optimization plan
            plan = {
                "routing": {
                    "algorithm": self.routing_algorithm,
                    "avoid_areas": avoid_areas,
                    "priority_areas": high_priority_areas
                },
                "scheduling": {
                    "use_sensors": self.has_sensors,
                    "threshold": 0.7 if self.has_sensors else 0.0,
                    "max_fuel": max_fuel
                }
            }
            
            # Add to optimization history
            self.optimization_history.append({
                "timestamp": datetime.datetime.now().isoformat(),
                "constraints": constraints,
                "improvements": improvements,
                "metrics": new_metrics
            })
            
            # Keep history manageable
            if len(self.optimization_history) > 100:
                self.optimization_history = self.optimization_history[-100:]
            
            return {
                "status": "success",
                "message": "Waste collection optimization completed",
                "metrics": self.current_metrics,
                "improvements": improvements,
                "plan": plan,
                "execution_time": time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"Error in waste optimization: {e}")
            return {
                "status": "error",
                "message": str(e),
                "metrics": self.current_metrics,
                "execution_time": time.time() - start_time
            }


class EmergencySubsystem(Subsystem):
    """Emergency response subsystem - controls emergency vehicle routing and dispatch."""
    
    def __init__(self, system_id: str, config: dict = None):
        super().__init__(system_id, "emergency", config)
        
        # Emergency-specific attributes
        self.routing_algorithm = self.config.get("routing_algorithm", "a_star")
        self.vehicle_tracking = self.config.get("vehicle_tracking", True)
        self.predictive_dispatch = self.config.get("predictive_dispatch", False)
        
        # Default metrics
        self.current_metrics = {
            "response_time": 310.0,  # seconds
            "coverage": 0.86,  # 0-1 scale
            "resource_utilization": 0.65,  # 0-1 scale
            "incidents_active": 3,  # count
            "vehicles_available": 12,  # count
            "critical_incidents": 1  # count
        }
    
    async def optimize(self, constraints: dict = None) -> dict:
        """Run emergency response optimization with given constraints."""
        start_time = time.time()
        constraints = constraints or {}
        
        try:
            # Use the API if available
            if self.api_endpoint:
                # Implementation for real API call would go here
                pass
            
            # Simulated optimization for demonstration
            # In a real implementation, this would run an emergency routing algorithm
            
            # Apply scenario constraints
            is_emergency = constraints.get("emergency_mode", False)
            restricted_zones = constraints.get("restricted_zones", [])
            incident_list = constraints.get("incidents", [])
            
            # Generate improvement metrics
            # In a real implementation, these would come from the optimization algorithm
            improvements = {
                "response_time_reduction": 45.0 if is_emergency else 20.0,
                "coverage_increase": 0.06 if is_emergency else 0.03,
                "utilization_increase": 0.15 if is_emergency else 0.05
            }
            
            # Update metrics
            new_metrics = {
                "response_time": max(180.0, self.current_metrics["response_time"] - improvements["response_time_reduction"]),
                "coverage": min(0.98, self.current_metrics["coverage"] + improvements["coverage_increase"]),
                "resource_utilization": min(0.95, self.current_metrics["resource_utilization"] + improvements["utilization_increase"]),
                "incidents_active": self.current_metrics["incidents_active"] + (len(incident_list) if incident_list else 0),
                "vehicles_available": max(2, self.current_metrics["vehicles_available"] - (1 if is_emergency else 0)),
                "critical_incidents": self.current_metrics["critical_incidents"] + (1 if is_emergency else 0)
            }
            
            self.current_metrics = new_metrics
            self.last_update = datetime.datetime.now()
            
            # Generate optimization plan
            plan = {
                "routing": {
                    "algorithm": self.routing_algorithm,
                    "avoid_zones": restricted_zones,
                    "emergency_override": is_emergency
                },
                "dispatch": {
                    "vehicle_tracking": self.vehicle_tracking,
                    "predictive": self.predictive_dispatch,
                    "incidents": incident_list
                }
            }
            
            # Add to optimization history
            self.optimization_history.append({
                "timestamp": datetime.datetime.now().isoformat(),
                "constraints": constraints,
                "improvements": improvements,
                "metrics": new_metrics
            })
            
            # Keep history manageable
            if len(self.optimization_history) > 100:
                self.optimization_history = self.optimization_history[-100:]
            
            return {
                "status": "success",
                "message": "Emergency response optimization completed",
                "metrics": self.current_metrics,
                "improvements": improvements,
                "plan": plan,
                "execution_time": time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"Error in emergency optimization: {e}")
            return {
                "status": "error",
                "message": str(e),
                "metrics": self.current_metrics,
                "execution_time": time.time() - start_time
            }


class CityScenario:
    """
    Represents a city-wide scenario that affects multiple systems.
    
    Examples include emergencies, weather events, planned closures, 
    or special events like concerts or sports games.
    """
    
    def __init__(self, scenario_id: str, name: str, description: str = None, 
                 priority: str = "medium", active: bool = False):
        """
        Initialize a city scenario.
        
        Args:
            scenario_id: Unique identifier
            name: Scenario name
            description: Detailed description
            priority: Priority level ('emergency', 'high', 'medium', 'low')
            active: Whether the scenario is currently active
        """
        self.id = scenario_id
        self.name = name
        self.description = description or name
        self.priority = priority
        self.active = active
        self.created_at = datetime.datetime.now()
        self.activated_at = None
        self.resolved_at = None
        
        # Impact on different subsystems (0-1 scale)
        self.impacts = {
            "traffic": 0.0,
            "energy": 0.0,
            "waste": 0.0,
            "emergency": 0.0
        }
        
        # Constraints for each subsystem
        self.constraints = {
            "traffic": {},
            "energy": {},
            "waste": {},
            "emergency": {}
        }
    
    def activate(self) -> None:
        """Activate this scenario."""
        self.active = True
        self.activated_at = datetime.datetime.now()
        logger.info(f"Activated scenario: {self.name}")
    
    def resolve(self) -> None:
        """Resolve (deactivate) this scenario."""
        self.active = False
        self.resolved_at = datetime.datetime.now()
        logger.info(f"Resolved scenario: {self.name}")
    
    def set_impact(self, subsystem_type: str, impact_level: float) -> None:
        """
        Set the impact level for a subsystem.
        
        Args:
            subsystem_type: Type of subsystem ('traffic', 'energy', etc.)
            impact_level: Impact level (0-1 scale)
        """
        if subsystem_type in self.impacts:
            self.impacts[subsystem_type] = max(0.0, min(1.0, impact_level))
    
    def add_constraint(self, subsystem_type: str, key: str, value: Any) -> None:
        """
        Add a constraint for a subsystem.
        
        Args:
            subsystem_type: Type of subsystem
            key: Constraint key
            value: Constraint value
        """
        if subsystem_type in self.constraints:
            self.constraints[subsystem_type][key] = value
    
    def get_priority_value(self) -> int:
        """Get numeric priority value."""
        return PRIORITY_LEVELS.get(self.priority, 50)
    
    def to_dict(self) -> dict:
        """
        Convert to dictionary representation.
        
        Returns:
            Dictionary with scenario data
        """
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "priority": self.priority,
            "active": self.active,
            "created_at": self.created_at.isoformat(),
            "activated_at": self.activated_at.isoformat() if self.activated_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "impacts": self.impacts,
            "constraints": self.constraints
        }


class Orchestrator:
    """
    Main orchestrator that coordinates all city subsystems.
    
    Manages the interactions between subsystems, resolves conflicts,
    and ensures global optimization instead of local optimization.
    """
    
    def __init__(self, config_file: str = None):
        """
        Initialize the orchestrator.
        
        Args:
            config_file: Path to configuration file
        """
        # Load configuration if provided
        self.config = {}
        if config_file:
            self._load_config(config_file)
        
        # Initialize collections
        self.subsystems = {}  # Dict[str, Subsystem]
        self.scenarios = {}  # Dict[str, CityScenario]
        self.active_scenarios = []  # List[str]
        
        # Initialize optimization weights
        self.weights = self.config.get("weights", DEFAULT_WEIGHTS.copy())
        
        # Optimization history
        self.optimization_history = []
        self.last_global_optimization = None
        
        # Background processing
        self.running = False
        self.should_stop = False
        self.update_thread = None
        self.optimization_thread = None
        
        # Event loop for async operations
        self.loop = asyncio.new_event_loop()
        
        logger.info("City Orchestrator initialized")
        
        # Auto-initialize subsystems if configured
        if self.config.get("auto_init", False):
            self._init_default_subsystems()
    
    def _load_config(self, config_file: str) -> None:
        """
        Load configuration from file.
        
        Args:
            config_file: Path to configuration file
        """
        try:
            with open(config_file, 'r') as f:
                if config_file.endswith('.json'):
                    self.config = json.load(f)
                elif config_file.endswith('.yaml') or config_file.endswith('.yml'):
                    self.config = yaml.safe_load(f)
                else:
                    logger.warning(f"Unknown config format: {config_file}")
            
            logger.info(f"Loaded configuration from {config_file}")
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            # Continue with empty config
    
    def _init_default_subsystems(self) -> None:
        """Initialize default subsystems from configuration."""
        subsystem_configs = self.config.get("subsystems", {})
        
        # Add traffic subsystem
        if "traffic" in subsystem_configs:
            self.add_subsystem(TrafficSubsystem("traffic_main", subsystem_configs["traffic"]))
        
        # Add energy subsystem
        if "energy" in subsystem_configs:
            self.add_subsystem(EnergySubsystem("energy_main", subsystem_configs["energy"]))
        
        # Add waste subsystem
        if "waste" in subsystem_configs:
            self.add_subsystem(WasteSubsystem("waste_main", subsystem_configs["waste"]))
        
        # Add emergency subsystem
        if "emergency" in subsystem_configs:
            self.add_subsystem(EmergencySubsystem("emergency_main", subsystem_configs["emergency"]))
        
        logger.info(f"Initialized {len(self.subsystems)} subsystems from configuration")
    
    def add_subsystem(self, subsystem: Subsystem) -> None:
        """
        Add a subsystem to the orchestrator.
        
        Args:
            subsystem: Subsystem instance
        """
        self.subsystems[subsystem.id] = subsystem
        logger.info(f"Added {subsystem.type} subsystem: {subsystem.id}")
    
    def remove_subsystem(self, subsystem_id: str) -> bool:
        """
        Remove a subsystem from the orchestrator.
        
        Args:
            subsystem_id: ID of subsystem to remove
            
        Returns:
            True if removed, False if not found
        """
        if subsystem_id in self.subsystems:
            del self.subsystems[subsystem_id]
            logger.info(f"Removed subsystem: {subsystem_id}")
            return True
        return False
    
    def add_scenario(self, scenario: CityScenario) -> None:
        """
        Add a scenario to the orchestrator.
        
        Args:
            scenario: Scenario instance
        """
        self.scenarios[scenario.id] = scenario
        if scenario.active:
            self.active_scenarios.append(scenario.id)
        logger.info(f"Added scenario: {scenario.name}")
    
    def activate_scenario(self, scenario_id: str) -> bool:
        """
        Activate a scenario.
        
        Args:
            scenario_id: ID of scenario to activate
            
        Returns:
            True if activated, False if not found
        """
        if scenario_id in self.scenarios:
            scenario = self.scenarios[scenario_id]
            scenario.activate()
            
            if scenario_id not in self.active_scenarios:
                self.active_scenarios.append(scenario_id)
            
            # Run a global optimization to respond to the scenario
            self.loop.run_until_complete(self.optimize())
            
            logger.info(f"Activated scenario: {scenario.name}")
            return True
        return False
    
    def resolve_scenario(self, scenario_id: str) -> bool:
        """
        Resolve (deactivate) a scenario.
        
        Args:
            scenario_id: ID of scenario to resolve
            
        Returns:
            True if resolved, False if not found
        """
        if scenario_id in self.scenarios:
            scenario = self.scenarios[scenario_id]
            scenario.resolve()
            
            if scenario_id in self.active_scenarios:
                self.active_scenarios.remove(scenario_id)
            
            # Run a global optimization to return to normal operations
            self.loop.run_until_complete(self.optimize())
            
            logger.info(f"Resolved scenario: {scenario.name}")
            return True
        return False
    
    def update_weights(self, new_weights: dict) -> None:
        """
        Update optimization weights.
        
        Args:
            new_weights: Dictionary of new weights
        """
        if not new_weights:
            return
        
        # Validate and normalize weights
        total = sum(new_weights.values())
        if total <= 0:
            logger.error("Invalid weights: sum must be positive")
            return
        
        normalized = {k: v/total for k, v in new_weights.items()}
        self.weights.update(normalized)
        logger.info(f"Updated optimization weights: {self.weights}")
    
    async def get_all_statuses(self) -> dict:
        """
        Get status of all subsystems.
        
        Returns:
            Dictionary with status information
        """
        statuses = {}
        
        # Get status for each subsystem concurrently
        tasks = []
        for subsystem_id, subsystem in self.subsystems.items():
            tasks.append(subsystem.get_status())
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                subsystem_id = list(self.subsystems.keys())[i]
                if isinstance(result, Exception):
                    logger.error(f"Error getting status for {subsystem_id}: {result}")
                    statuses[subsystem_id] = {
                        "status": "error",
                        "message": str(result)
                    }
                else:
                    statuses[subsystem_id] = result
        
        return statuses
    
    def _get_global_constraints(self) -> dict:
        """
        Calculate global constraints from active scenarios.
        
        Returns:
            Dictionary of global constraints
        """
        global_constraints = {}
        
        # Sort scenarios by priority (highest first)
        active_scenarios = [self.scenarios[sid] for sid in self.active_scenarios if sid in self.scenarios]
        active_scenarios.sort(key=lambda s: s.get_priority_value(), reverse=True)
        
        for scenario in active_scenarios:
            # Extract key constraints based on scenario type
            if "emergency" in scenario.name.lower():
                # Emergency scenarios take precedence
                global_constraints["emergency_mode"] = True
                global_constraints["priority_response"] = True
            
            elif "weather" in scenario.name.lower():
                # Weather scenarios
                global_constraints["weather_response"] = True
                global_constraints["safety_priority"] = True
                
                # Get weather type if specified
                for weather_key in ["storm", "snow", "flood", "heat"]:
                    if weather_key in scenario.name.lower():
                        global_constraints["weather_type"] = weather_key
            
            elif "event" in scenario.name.lower():
                # Special events
                global_constraints["event_management"] = True
                global_constraints["crowd_management"] = True
            
            # Add any restricted areas
            for subsystem_type in ["traffic", "waste", "emergency"]:
                area_key = "restricted_areas" if subsystem_type != "emergency" else "restricted_zones"
                
                if area_key in scenario.constraints.get(subsystem_type, {}):
                    areas = scenario.constraints[subsystem_type][area_key]
                    if area_key not in global_constraints:
                        global_constraints[area_key] = []
                    global_constraints[area_key].extend(areas)
        
        return global_constraints
    
    def _resolve_constraint_conflicts(self, constraints_by_subsystem: dict) -> dict:
        """
        Resolve conflicts between constraints for different subsystems.
        
        Args:
            constraints_by_subsystem: Dictionary of constraints by subsystem
            
        Returns:
            Dictionary of resolved constraints
        """
        resolved = {}
        
        # Identify conflicts and resolve them
        # For simplicity, we'll prioritize emergency > energy > traffic > waste
        priority_order = ["emergency", "energy", "traffic", "waste"]
        
        # Process each subsystem type in priority order
        for subsystem_type in priority_order:
            if subsystem_type not in constraints_by_subsystem:
                continue
                
            subsystem_constraints = constraints_by_subsystem[subsystem_type]
            resolved[subsystem_type] = {}
            
            for key, value in subsystem_constraints.items():
                # Check for conflicts with higher priority subsystems
                conflict = False
                
                # Simple conflict resolution by avoiding direct contradictions
                # In a real system, this would need more sophisticated logic
                if key == "max_fuel" and "emergency_mode" in self._get_global_constraints():
                    # Don't limit fuel during emergencies
                    conflict = True
                
                if not conflict:
                    resolved[subsystem_type][key] = value
        
        return resolved
    
    async def optimize_subsystem(self, subsystem_id: str, constraints: dict = None) -> dict:
        """
        Optimize a specific subsystem.
        
        Args:
            subsystem_id: ID of subsystem to optimize
            constraints: Optional constraints
            
        Returns:
            Optimization results
        """
        if subsystem_id not in self.subsystems:
            return {
                "status": "error",
                "message": f"Subsystem not found: {subsystem_id}"
            }
        
        subsystem = self.subsystems[subsystem_id]
        logger.info(f"Optimizing subsystem: {subsystem_id}")
        
        try:
            # Set a timeout for optimization
            result = await asyncio.wait_for(
                subsystem.optimize(constraints),
                timeout=MAX_EXECUTION_TIME
            )
            
            return result
            
        except asyncio.TimeoutError:
            logger.error(f"Optimization timed out for {subsystem_id}")
            return {
                "status": "error",
                "message": f"Optimization timed out after {MAX_EXECUTION_TIME}s"
            }
        except Exception as e:
            logger.error(f"Error optimizing {subsystem_id}: {e}")
            return {
                "status": "error",
                "message": f"Optimization error: {str(e)}"
            }
    
    async def optimize(self) -> dict:
        """
        Run a global optimization across all subsystems.
        
        This is the core orchestration function that coordinates all subsystems.
        
        Returns:
            Dictionary with optimization results
        """
        start_time = time.time()
        logger.info("Starting global optimization")
        
        try:
            # Get current status of all subsystems
            statuses = await self.get_all_statuses()
            
            # Calculate global constraints from active scenarios
            global_constraints = self._get_global_constraints()
            logger.info(f"Global constraints: {global_constraints}")
            
            # Build dependency graph for optimization order
            graph = nx.DiGraph()
            
            # Add all subsystems as nodes
            for subsystem_id in self.subsystems:
                graph.add_node(subsystem_id)
            
            # Add dependencies as edges
            for subsystem_id, subsystem in self.subsystems.items():
                for dependency_id in subsystem.dependencies:
                    if dependency_id in self.subsystems:
                        graph.add_edge(dependency_id, subsystem_id)
            
            # Determine optimization order (topological sort)
            try:
                optimization_order = list(nx.topological_sort(graph))
                logger.info(f"Optimization order: {optimization_order}")
            except nx.NetworkXUnfeasible:
                # Cycle in dependencies, use a priority order instead
                logger.warning("Dependency cycle detected, using priority order")
                type_priority = {
                    "emergency": 0,
                    "energy": 1,
                    "traffic": 2,
                    "waste": 3
                }
                optimization_order = sorted(
                    self.subsystems.keys(),
                    key=lambda s_id: type_priority.get(self.subsystems[s_id].type, 99)
                )
            
            # Calculate subsystem-specific constraints
            subsystem_constraints = {}
            
            for subsystem_id in optimization_order:
                subsystem = self.subsystems[subsystem_id]
                constraints = {}
                
                # Add global constraints relevant to this subsystem
                if subsystem.type == "traffic":
                    if global_constraints.get("emergency_mode"):
                        constraints["emergency_mode"] = True
                    if global_constraints.get("restricted_areas"):
                        constraints["restricted_areas"] = global_constraints["restricted_areas"]
                    if global_constraints.get("weather_response"):
                        constraints["weather_conditions"] = global_constraints.get("weather_type", "adverse")
                
                elif subsystem.type == "energy":
                    if global_constraints.get("emergency_mode"):
                        constraints["critical_services"] = ["emergency", "hospital", "traffic_signals"]
                    if global_constraints.get("weather_type") == "storm":
                        constraints["storm_protection"] = True
                
                elif subsystem.type == "waste":
                    if global_constraints.get("restricted_areas"):
                        constraints["avoid_areas"] = global_constraints["restricted_areas"]
                    if global_constraints.get("emergency_mode"):
                        constraints["delay_collection"] = True
                
                elif subsystem.type == "emergency":
                    if global_constraints.get("emergency_mode"):
                        constraints["emergency_mode"] = True
                    if global_constraints.get("restricted_zones"):
                        constraints["restricted_zones"] = global_constraints["restricted_zones"]
                
                # Add scenario-specific constraints
                for scenario_id in self.active_scenarios:
                    if scenario_id in self.scenarios:
                        scenario = self.scenarios[scenario_id]
                        if subsystem.type in scenario.constraints:
                            # Add each constraint from the scenario
                            for key, value in scenario.constraints[subsystem.type].items():
                                constraints[key] = value
                
                subsystem_constraints[subsystem_id] = constraints
            
            # Resolve any conflicts between constraints
            resolved_constraints = self._resolve_constraint_conflicts(subsystem_constraints)
            
            # Run optimization for each subsystem in order
            optimization_results = {}
            
            for subsystem_id in optimization_order:
                constraints = resolved_constraints.get(subsystem_id, {})
                result = await self.optimize_subsystem(subsystem_id, constraints)
                optimization_results[subsystem_id] = result
            
            # Calculate global objective score
            global_score = self._calculate_global_score(optimization_results)
            
            # Apply optimization plans in priority order
            application_results = {}
            
            # Priority order for application (always apply emergency first)
            application_order = sorted(
                optimization_results.keys(),
                key=lambda s_id: 0 if self.subsystems[s_id].type == "emergency" else 1
            )
            
            for subsystem_id in application_order:
                result = optimization_results[subsystem_id]
                
                # Only apply successful optimizations with a plan
                if result.get("status") == "success" and "plan" in result:
                    application_result = await self.subsystems[subsystem_id].apply_optimization(result["plan"])
                    application_results[subsystem_id] = application_result
            
            # Update last optimization time
            self.last_global_optimization = datetime.datetime.now()
            
            # Store optimization in history
            optimization_record = {
                "timestamp": self.last_global_optimization.isoformat(),
                "global_constraints": global_constraints,
                "active_scenarios": self.active_scenarios,
                "global_score": global_score,
                "execution_time": time.time() - start_time,
                "results": {s_id: result.get("status") for s_id, result in optimization_results.items()}
            }
            
            self.optimization_history.append(optimization_record)
            
            # Limit history size
            if len(self.optimization_history) > 100:
                self.optimization_history = self.optimization_history[-100:]
            
            logger.info(f"Global optimization completed in {time.time() - start_time:.2f}s with score {global_score:.4f}")
            
            return {
                "status": "success",
                "message": "Global optimization completed",
                "global_score": global_score,
                "global_constraints": global_constraints,
                "active_scenarios": [self.scenarios[s_id].name for s_id in self.active_scenarios if s_id in self.scenarios],
                "optimization_results": optimization_results,
                "application_results": application_results,
                "execution_time": time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"Error in global optimization: {e}")
            return {
                "status": "error",
                "message": f"Global optimization failed: {str(e)}"
            }
    
    def _calculate_global_score(self, optimization_results: dict) -> float:
        """
        Calculate a global optimization score.
        
        Args:
            optimization_results: Dictionary of optimization results by subsystem
            
        Returns:
            Global optimization score (0-1 scale)
        """
        score = 0.0
        
        # Extract metrics from each subsystem
        for subsystem_id, result in optimization_results.items():
            if result.get("status") != "success" or "metrics" not in result:
                continue
                
            subsystem = self.subsystems[subsystem_id]
            metrics = result["metrics"]
            subsystem_score = 0.0
            
            # Calculate subsystem-specific score
            if subsystem.type == "traffic":
                subsystem_score = (
                    (1.0 - metrics.get("congestion_level", 0.5)) * 0.4 +
                    (metrics.get("average_speed", 30) / 50) * 0.3 +
                    (1.0 - metrics.get("average_wait_time", 45) / 120) * 0.3
                )
            
            elif subsystem.type == "energy":
                subsystem_score = (
                    metrics.get("grid_stability", 0.8) * 0.3 +
                    (metrics.get("renewable_percentage", 20) / 100) * 0.4 +
                    (1.0 - metrics.get("carbon_intensity", 400) / 1000) * 0.3
                )
            
            elif subsystem.type == "waste":
                subsystem_score = (
                    metrics.get("collection_efficiency", 0.7) * 0.5 +
                    (1.0 - metrics.get("average_fill_level", 0.6)) * 0.3 +
                    (1.0 - metrics.get("overflow_incidents", 5) / 20) * 0.2
                )
            
            elif subsystem.type == "emergency":
                # For emergency, lower response time is better
                response_time_score = 1.0 - min(1.0, metrics.get("response_time", 300) / 600)
                
                subsystem_score = (
                    response_time_score * 0.5 +
                    metrics.get("coverage", 0.8) * 0.3 +
                    metrics.get("resource_utilization", 0.7) * 0.2
                )
            
            # Apply weight for this subsystem type
            weight = self.weights.get(subsystem.type, 0.25)
            score += subsystem_score * weight
        
        return score
    
    def start(self) -> None:
        """Start the orchestrator's background processes."""
        if self.running:
            logger.warning("Orchestrator is already running")
            return
        
        self.running = True
        self.should_stop = False
        
        # Run initial optimization
        self.loop.run_until_complete(self.optimize())
        
        # Start background threads
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        
        self.optimization_thread = threading.Thread(target=self._optimization_loop)
        self.optimization_thread.daemon = True
        self.optimization_thread.start()
        
        logger.info("Orchestrator started")
    
    def stop(self) -> None:
        """Stop the orchestrator's background processes."""
        if not self.running:
            logger.warning("Orchestrator is not running")
            return
        
        self.should_stop = True
        self.running = False
        
        # Wait for threads to finish (with timeout)
        if self.update_thread:
            self.update_thread.join(timeout=2.0)
        
        if self.optimization_thread:
            self.optimization_thread.join(timeout=2.0)
        
        logger.info("Orchestrator stopped")
    
    def _update_loop(self) -> None:
        """Background thread for periodic status updates."""
        next_update = time.time()
        
        while not self.should_stop:
            current_time = time.time()
            
            if current_time >= next_update:
                # Run status update
                asyncio.run(self.get_all_statuses())
                
                # Schedule next update
                next_update = current_time + STATUS_UPDATE_INTERVAL
            
            # Sleep a bit to avoid busy waiting
            time.sleep(1.0)
    
    def _optimization_loop(self) -> None:
        """Background thread for periodic optimization."""
        next_optimization = time.time() + OPTIMIZATION_INTERVAL
        
        while not self.should_stop:
            current_time = time.time()
            
            if current_time >= next_optimization:
                # Run optimization
                asyncio.run(self.optimize())
                
                # Schedule next optimization
                next_optimization = current_time + OPTIMIZATION_INTERVAL
            
            # Sleep a bit to avoid busy waiting
            time.sleep(1.0)
    
    def get_system_status(self) -> dict:
        """
        Get the current status of the entire system.
        
        Returns:
            Dictionary with status information
        """
        # Get subsystem statuses
        subsystem_statuses = {}
        for subsystem_id, subsystem in self.subsystems.items():
            subsystem_statuses[subsystem_id] = {
                "type": subsystem.type,
                "status": subsystem.status,
                "last_update": subsystem.last_update.isoformat(),
                "metrics": subsystem.current_metrics
            }
        
        # Get active scenarios
        active_scenario_details = []
        for scenario_id in self.active_scenarios:
            if scenario_id in self.scenarios:
                active_scenario_details.append(self.scenarios[scenario_id].to_dict())
        
        return {
            "status": "running" if self.running else "stopped",
            "last_optimization": self.last_global_optimization.isoformat() if self.last_global_optimization else None,
            "subsystems": subsystem_statuses,
            "active_scenarios": active_scenario_details,
            "optimization_weights": self.weights
        }
    
    def save_state(self, filepath: str) -> bool:
        """
        Save orchestrator state to a file.
        
        Args:
            filepath: Path to save file
            
        Returns:
            True if successful
        """
        try:
            # Create a serializable state object
            state = {
                "config": self.config,
                "weights": self.weights,
                "last_optimization": self.last_global_optimization.isoformat() if self.last_global_optimization else None,
                "optimization_history": self.optimization_history,
                # Don't serialize the actual subsystem objects
                "subsystems": {
                    s_id: {
                        "type": s.type,
                        "config": {k: v for k, v in s.config.items() if k != "api_key"},
                        "current_metrics": s.current_metrics,
                        "dependencies": s.dependencies
                    } for s_id, s in self.subsystems.items()
                },
                "scenarios": {
                    s_id: s.to_dict() for s_id, s in self.scenarios.items()
                },
                "active_scenarios": self.active_scenarios
            }
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            
            # Write to file
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)
                
            logger.info(f"Saved orchestrator state to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving state: {e}")
            return False
    
    def load_state(self, filepath: str) -> bool:
        """
        Load orchestrator state from a file.
        
        Args:
            filepath: Path to load file from
            
        Returns:
            True if successful
        """
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            # Restore configuration and weights
            self.config = state.get("config", {})
            self.weights = state.get("weights", DEFAULT_WEIGHTS.copy())
            
            # Restore last optimization time
            if state.get("last_optimization"):
                try:
                    self.last_global_optimization = datetime.datetime.fromisoformat(state["last_optimization"])
                except ValueError:
                    self.last_global_optimization = None
            
            # Restore optimization history
            self.optimization_history = state.get("optimization_history", [])
            
            # Restore scenarios
            self.scenarios = {}
            for s_id, s_data in state.get("scenarios", {}).items():
                scenario = CityScenario(
                    s_data["id"], 
                    s_data["name"],
                    s_data.get("description"),
                    s_data.get("priority", "medium"),
                    s_data.get("active", False)
                )
                
                # Restore impact levels
                for subsystem_type, impact in s_data.get("impacts", {}).items():
                    scenario.set_impact(subsystem_type, impact)
                
                # Restore constraints
                for subsystem_type, constraints in s_data.get("constraints", {}).items():
                    for key, value in constraints.items():
                        scenario.add_constraint(subsystem_type, key, value)
                
                # Restore timestamps
                if s_data.get("created_at"):
                    try:
                        scenario.created_at = datetime.datetime.fromisoformat(s_data["created_at"])
                    except ValueError:
                        pass
                
                if s_data.get("activated_at"):
                    try:
                        scenario.activated_at = datetime.datetime.fromisoformat(s_data["activated_at"])
                    except ValueError:
                        pass
                
                if s_data.get("resolved_at"):
                    try:
                        scenario.resolved_at = datetime.datetime.fromisoformat(s_data["resolved_at"])
                    except ValueError:
                        pass
                
                self.scenarios[s_id] = scenario
            
            # Restore active scenarios
            self.active_scenarios = state.get("active_scenarios", [])
            
            # Recreate subsystems
            self.subsystems = {}
            for s_id, s_data in state.get("subsystems", {}).items():
                subsystem_type = s_data["type"]
                config = s_data.get("config", {})
                
                # Create appropriate subsystem class
                if subsystem_type == "traffic":
                    subsystem = TrafficSubsystem(s_id, config)
                elif subsystem_type == "energy":
                    subsystem = EnergySubsystem(s_id, config)
                elif subsystem_type == "waste":
                    subsystem = WasteSubsystem(s_id, config)
                elif subsystem_type == "emergency":
                    subsystem = EmergencySubsystem(s_id, config)
                else:
                    subsystem = Subsystem(s_id, subsystem_type, config)
                
                # Restore metrics and dependencies
                subsystem.current_metrics = s_data.get("current_metrics", {})
                subsystem.dependencies = s_data.get("dependencies", [])
                
                self.subsystems[s_id] = subsystem
            
            logger.info(f"Loaded orchestrator state from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading state: {e}")
            return False


# --- Example scenarios ---

def create_emergency_scenario() -> CityScenario:
    """Create an example emergency scenario."""
    scenario = CityScenario(
        "emergency1",
        "Major Traffic Accident",
        "Multi-vehicle accident on highway with injuries",
        "emergency"
    )
    
    # Set impact levels
    scenario.set_impact("traffic", 0.9)
    scenario.set_impact("emergency", 0.9)
    scenario.set_impact("energy", 0.2)
    scenario.set_impact("waste", 0.3)
    
    # Add constraints
    scenario.add_constraint("traffic", "restricted_areas", [
        {"lat": 37.7749, "lon": -122.4194, "radius": 500}  # Example location
    ])
    scenario.add_constraint("traffic", "emergency_mode", True)
    
    scenario.add_constraint("emergency", "incidents", [
        {"id": "incident1", "type": "accident", "priority": 1, 
         "location": {"lat": 37.7749, "lon": -122.4194}}
    ])
    
    return scenario


def create_weather_scenario() -> CityScenario:
    """Create an example weather scenario."""
    scenario = CityScenario(
        "weather1",
        "Severe Storm Warning",
        "Heavy rain and wind affecting downtown area",
        "high"
    )
    
    # Set impact levels
    scenario.set_impact("traffic", 0.7)
    scenario.set_impact("energy", 0.8)
    scenario.set_impact("waste", 0.6)
    scenario.set_impact("emergency", 0.5)
    
    # Add constraints
    scenario.add_constraint("traffic", "weather_conditions", "storm")
    scenario.add_constraint("energy", "storm_protection", True)
    scenario.add_constraint("waste", "delay_collection", ["downtown", "north_district"])
    
    return scenario


# --- Main function ---

def run_orchestrator(config_file: str = None, load_file: str = None, save_file: str = None):
    """
    Run the city orchestrator.
    
    Args:
        config_file: Path to configuration file
        load_file: Path to load state from
        save_file: Path to save state to
    """
    # Create orchestrator
    orchestrator = Orchestrator(config_file)
    
    # Load state if specified
    if load_file and os.path.exists(load_file):
        if orchestrator.load_state(load_file):
            print(f"Loaded state from {load_file}")
        else:
            print(f"Failed to load state from {load_file}")
    
    # Add sample scenarios and subsystems if none exist
    if not orchestrator.scenarios:
        emergency = create_emergency_scenario()
        weather = create_weather_scenario()
        orchestrator.add_scenario(emergency)
        orchestrator.add_scenario(weather)
        print("Added sample scenarios")
    
    if not orchestrator.subsystems:
        # Add example subsystems
        orchestrator.add_subsystem(TrafficSubsystem("traffic_main"))
        orchestrator.add_subsystem(EnergySubsystem("energy_main"))
        orchestrator.add_subsystem(WasteSubsystem("waste_main"))
        orchestrator.add_subsystem(EmergencySubsystem("emergency_main"))
        print("Added sample subsystems")
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        print("\nShutting down orchestrator...")
        orchestrator.stop()
        
        if save_file:
            if orchestrator.save_state(save_file):
                print(f"Saved state to {save_file}")
            else:
                print(f"Failed to save state to {save_file}")
                
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start orchestrator
    orchestrator.start()
    print("City AI Orchestrator is running. Press Ctrl+C to stop.")
    
    # Activate a scenario for testing
    if orchestrator.scenarios and not orchestrator.active_scenarios:
        first_id = next(iter(orchestrator.scenarios.keys()))
        if orchestrator.activate_scenario(first_id):
            print(f"Activated scenario: {orchestrator.scenarios[first_id].name}")
    
    # Keep main thread alive
    try:
        while orchestrator.running:
            time.sleep(1)
    except KeyboardInterrupt:
        signal_handler(None, None)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Smart City AI Orchestrator")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--load", help="Load state from file")
    parser.add_argument("--save", help="Save state to file")
    
    args = parser.parse_args()
    
    run_orchestrator(args.config, args.load, args.save)

"""
SUMMARY:
========
This module implements a Smart City AI Orchestrator that coordinates multiple subsystems
(traffic, energy, waste, emergency) to ensure they work together efficiently. It handles
city-wide scenarios like emergencies and storms, resolves conflicting priorities, and 
optimizes resource allocation across all domains.

Key components:
1. Subsystem classes for each city domain (traffic, energy, waste, emergency)
2. CityScenario class for representing events affecting multiple systems
3. Orchestrator class that manages coordination and optimization
4. Background processing for continuous monitoring and optimization

The orchestrator uses weighted multi-objective optimization, dependency-based execution
ordering, and constraint resolution to achieve global rather than local optimization.

TODO:
=====
1. Add machine learning model for predictive optimization
2. Implement more sophisticated conflict resolution algorithms
3. Create visualization dashboard for monitoring system performance
4. Add interfaces to real city infrastructure APIs
5. Implement anomaly detection for early crisis identification
6. Add citizen feedback integration for optimization goals
7. Create more detailed simulation models for testing scenarios
8. Implement federated learning across multiple city nodes
9. Develop dynamic re-optimization based on real-time events
10. Add sustainability and carbon footprint metrics
11. Create adaptable optimization weights based on outcomes
12. Implement notification system for key stakeholders
13. Add multi-city coordination capabilities
"""
