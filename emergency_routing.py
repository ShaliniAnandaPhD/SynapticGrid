#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Emergency Response Routing
=========================
Uses AI to optimize emergency vehicle routes, avoiding congestion and delays.
This system interfaces with traffic data, road closures, and historical response
times to find optimal routes for emergency vehicles.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from shapely.geometry import Point, LineString
import matplotlib.pyplot as plt
import osmnx as ox
import folium
from folium.plugins import MarkerCluster
import requests
import json
import time
import logging
import os
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union, Set
from collections import defaultdict
import heapq
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("emergency_routing.log"), logging.StreamHandler()]
)
logger = logging.getLogger("EmergencyRouting")

# Constants
DEFAULT_CONGESTION_WEIGHT = 3.0  # How much to penalize congested roads
DEFAULT_SPEED_AMBULANCE = 80  # km/h on open roads
DEFAULT_SPEED_FIRETRUCK = 65  # km/h on open roads
DEFAULT_SPEED_POLICE = 90  # km/h on open roads
HISTORICAL_DATA_WINDOW = 7  # days of historical data to use
TRAFFIC_UPDATE_INTERVAL = 300  # seconds between traffic data updates
ROUTING_TIMEOUT = 5.0  # seconds allowed for routing computation
API_RETRY_ATTEMPTS = 3  # times to retry API calls


class EmergencyVehicle:
    """
    Represents an emergency vehicle with its capabilities and current status.
    
    This class tracks a vehicle's location, status, and routing capabilities.
    """
    
    def __init__(self, vehicle_id: str, vehicle_type: str, station_id: str, 
                capabilities: List[str] = None, location: Tuple[float, float] = None):
        """
        Initialize an emergency vehicle.
        
        Args:
            vehicle_id: Unique identifier for the vehicle
            vehicle_type: Type of vehicle ('ambulance', 'firetruck', 'police')
            station_id: ID of the home station
            capabilities: List of special capabilities this vehicle has
            location: Current (latitude, longitude) if not at station
        """
        self.id = vehicle_id
        self.type = vehicle_type
        self.station_id = station_id
        self.capabilities = capabilities or []
        self.location = location
        self.status = "available"  # available, dispatched, returning, maintenance
        self.current_incident_id = None
        self.route = []  # List of (lat, lon) points in current route
        self.route_time_estimate = 0  # Estimated time to arrival in seconds
        self.last_update_time = datetime.now()
        
        # Set speed and other parameters based on vehicle type
        if vehicle_type == "ambulance":
            self.max_speed = DEFAULT_SPEED_AMBULANCE
            self.acceleration = 3.5  # m/s²
            self.turning_radius = 7.5  # meters
        elif vehicle_type == "firetruck":
            self.max_speed = DEFAULT_SPEED_FIRETRUCK
            self.acceleration = 2.0  # m/s²
            self.turning_radius = 12.0  # meters
        elif vehicle_type == "police":
            self.max_speed = DEFAULT_SPEED_POLICE
            self.acceleration = 4.0  # m/s²
            self.turning_radius = 6.0  # meters
        else:
            # Default values for unknown vehicle types
            self.max_speed = 70  # km/h
            self.acceleration = 3.0  # m/s²
            self.turning_radius = 8.0  # meters
            
        logger.debug(f"Initialized {vehicle_type} {vehicle_id} at station {station_id}")
    
    def update_location(self, lat: float, lon: float, timestamp: datetime = None) -> None:
        """
        Update the vehicle's location.
        
        Args:
            lat: Latitude
            lon: Longitude
            timestamp: Time of the location update
        """
        self.location = (lat, lon)
        self.last_update_time = timestamp or datetime.now()
    
    def update_status(self, new_status: str, incident_id: str = None) -> None:
        """
        Update the vehicle's status.
        
        Args:
            new_status: New status string
            incident_id: ID of incident if being dispatched
        """
        old_status = self.status
        self.status = new_status
        
        if new_status == "dispatched":
            self.current_incident_id = incident_id
        elif new_status == "available":
            self.current_incident_id = None
            self.route = []
            self.route_time_estimate = 0
            
        logger.info(f"Vehicle {self.id} status changed from {old_status} to {new_status}" +
                  (f" for incident {incident_id}" if incident_id else ""))
    
    def set_route(self, route_points: List[Tuple[float, float]], time_estimate: float) -> None:
        """
        Set the vehicle's current route.
        
        Args:
            route_points: List of (lat, lon) waypoints
            time_estimate: Estimated travel time in seconds
        """
        self.route = route_points
        self.route_time_estimate = time_estimate
        logger.debug(f"Set route for vehicle {self.id} with {len(route_points)} points, " +
                   f"ETA: {time_estimate:.1f}s")
    
    def get_eta(self) -> float:
        """
        Get the estimated time of arrival in seconds.
        
        Returns:
            ETA in seconds, or 0 if not en route
        """
        if not self.route or self.status not in ["dispatched", "returning"]:
            return 0
            
        return self.route_time_estimate
    
    def to_dict(self) -> Dict:
        """
        Convert vehicle to dictionary representation.
        
        Returns:
            Dictionary of vehicle attributes
        """
        return {
            "id": self.id,
            "type": self.type,
            "station_id": self.station_id,
            "capabilities": self.capabilities,
            "location": self.location,
            "status": self.status,
            "current_incident_id": self.current_incident_id,
            "route_time_estimate": self.route_time_estimate,
            "last_update": self.last_update_time.isoformat()
        }


class EmergencyStation:
    """
    Represents an emergency services station (fire, police, ambulance).
    
    Tracks station location, vehicle assignments, and dispatch statistics.
    """
    
    def __init__(self, station_id: str, station_type: str, location: Tuple[float, float],
                address: str = None, capacity: int = 5):
        """
        Initialize an emergency station.
        
        Args:
            station_id: Unique identifier for the station
            station_type: Type of station ('fire', 'hospital', 'police', 'combined')
            location: (latitude, longitude) of the station
            address: Street address
            capacity: Maximum number of vehicles
        """
        self.id = station_id
        self.type = station_type
        self.location = location
        self.address = address
        self.capacity = capacity
        self.vehicles = []  # List of vehicle IDs assigned to this station
        
        # Track dispatch stats
        self.total_dispatches = 0
        self.avg_response_time = 0
        self.dispatches_by_hour = [0] * 24
        self.dispatches_by_type = defaultdict(int)
        
        logger.debug(f"Initialized {station_type} station {station_id} at {location}")
    
    def add_vehicle(self, vehicle_id: str) -> bool:
        """
        Add a vehicle to this station.
        
        Args:
            vehicle_id: ID of vehicle to add
            
        Returns:
            True if added successfully, False if station at capacity
        """
        if len(self.vehicles) >= self.capacity:
            logger.warning(f"Cannot add vehicle {vehicle_id} to station {self.id} - at capacity")
            return False
            
        self.vehicles.append(vehicle_id)
        logger.debug(f"Added vehicle {vehicle_id} to station {self.id}")
        return True
    
    def remove_vehicle(self, vehicle_id: str) -> bool:
        """
        Remove a vehicle from this station.
        
        Args:
            vehicle_id: ID of vehicle to remove
            
        Returns:
            True if removed successfully, False if vehicle not found
        """
        if vehicle_id in self.vehicles:
            self.vehicles.remove(vehicle_id)
            logger.debug(f"Removed vehicle {vehicle_id} from station {self.id}")
            return True
        
        return False
    
    def available_vehicles(self, vehicle_type: str = None) -> List[str]:
        """
        Get IDs of available vehicles at this station.
        
        Args:
            vehicle_type: Optional vehicle type filter
            
        Returns:
            List of vehicle IDs
        """
        # This method would work with the actual vehicle objects
        # but just returns the IDs for now - routing system handles the rest
        return self.vehicles
    
    def record_dispatch(self, vehicle_type: str, response_time: float,
                      incident_type: str) -> None:
        """
        Record statistics for a dispatch from this station.
        
        Args:
            vehicle_type: Type of vehicle dispatched
            response_time: Time in seconds from dispatch to arrival
            incident_type: Type of incident
        """
        # Update total dispatches
        self.total_dispatches += 1
        
        # Update average response time
        self.avg_response_time = ((self.avg_response_time * (self.total_dispatches - 1)) + 
                                response_time) / self.total_dispatches
        
        # Update hourly stats
        hour = datetime.now().hour
        self.dispatches_by_hour[hour] += 1
        
        # Update incident type stats
        self.dispatches_by_type[incident_type] += 1
    
    def to_dict(self) -> Dict:
        """
        Convert station to dictionary representation.
        
        Returns:
            Dictionary of station attributes
        """
        return {
            "id": self.id,
            "type": self.type,
            "location": self.location,
            "address": self.address,
            "capacity": self.capacity,
            "vehicles": self.vehicles,
            "total_dispatches": self.total_dispatches,
            "avg_response_time": self.avg_response_time,
            "dispatches_by_hour": self.dispatches_by_hour,
            "dispatches_by_type": dict(self.dispatches_by_type)
        }


class RoadNetwork:
    """
    Represents the city's road network for emergency routing.
    
    Maintains the graph structure, traffic conditions, and routing algorithms.
    """
    
    def __init__(self, city_name: str = None, bbox: Tuple[float, float, float, float] = None):
        """
        Initialize the road network.
        
        Args:
            city_name: Name of the city to load (if using OSM data)
            bbox: Bounding box (min_lat, min_lon, max_lat, max_lon) if not using city name
        """
        self.city_name = city_name
        self.bbox = bbox
        self.graph = None  # NetworkX graph of the road network
        self.nodes = {}  # Dict of node ID to (lat, lon)
        self.edges = {}  # Dict of (node1, node2) to edge attributes
        self.last_traffic_update = None
        
        # Edge attribute history for traffic patterns
        self.edge_history = {}  # Dict of (node1, node2) to list of (timestamp, travel_time)
        
        # Load the road network
        self._load_network()
        
        logger.info(f"Initialized road network for {city_name or 'custom area'}")
    
    def _load_network(self) -> None:
        """Load the road network from OSM or custom data."""
        try:
            if self.city_name:
                # Load from OSM by city name
                self.graph = ox.graph_from_place(self.city_name, network_type='drive')
                logger.info(f"Loaded road network for {self.city_name} from OSM")
            elif self.bbox:
                # Load from OSM by bounding box
                north, south, east, west = self.bbox
                self.graph = ox.graph_from_bbox(north, south, east, west, network_type='drive')
                logger.info(f"Loaded road network for bounding box from OSM")
            else:
                # Create an empty graph if no source specified
                self.graph = nx.MultiDiGraph()
                logger.warning("Created empty road network - no city or bbox specified")
            
            # Simplify the graph
            if self.graph:
                self.graph = ox.simplify_graph(self.graph)
                
                # Add travel time as edge weight
                self.graph = ox.add_edge_speeds(self.graph)
                self.graph = ox.add_edge_travel_times(self.graph)
                
                # Extract nodes and edges for easier access
                for node, data in self.graph.nodes(data=True):
                    self.nodes[node] = (data.get('y', 0), data.get('x', 0))
                
                for u, v, k, data in self.graph.edges(keys=True, data=True):
                    self.edges[(u, v, k)] = {
                        'length': data.get('length', 0),
                        'speed_limit': data.get('speed_kph', 50),
                        'travel_time': data.get('travel_time', 0),
                        'congestion': 0.0,  # 0-1 scale
                        'road_type': data.get('highway', 'residential')
                    }
        
        except Exception as e:
            logger.error(f"Error loading road network: {str(e)}")
            # Create an empty graph as fallback
            self.graph = nx.MultiDiGraph()
    
    def update_traffic(self, traffic_data: Dict) -> None:
        """
        Update traffic conditions from real-time data.
        
        Args:
            traffic_data: Dictionary mapping edge IDs to congestion values
        """
        updated = 0
        
        for edge_id, congestion in traffic_data.items():
            # Parse edge ID (format: "node1,node2,key")
            try:
                parts = edge_id.split(',')
                if len(parts) == 3:
                    u, v, k = int(parts[0]), int(parts[1]), int(parts[2])
                    edge_key = (u, v, k)
                    
                    if edge_key in self.edges:
                        # Update congestion value (0-1 scale)
                        self.edges[edge_key]['congestion'] = max(0, min(1, congestion))
                        
                        # Update travel time based on congestion
                        base_time = self.edges[edge_key]['length'] / (self.edges[edge_key]['speed_limit'] / 3.6)
                        congestion_factor = 1 + (congestion * DEFAULT_CONGESTION_WEIGHT)
                        self.edges[edge_key]['travel_time'] = base_time * congestion_factor
                        
                        # Store in history
                        if edge_key not in self.edge_history:
                            self.edge_history[edge_key] = []
                        
                        self.edge_history[edge_key].append((datetime.now(), self.edges[edge_key]['travel_time']))
                        
                        # Limit history size
                        if len(self.edge_history[edge_key]) > 1000:
                            self.edge_history[edge_key] = self.edge_history[edge_key][-1000:]
                            
                        updated += 1
            except Exception as e:
                logger.warning(f"Error updating traffic for edge {edge_id}: {str(e)}")
        
        self.last_traffic_update = datetime.now()
        logger.info(f"Updated traffic data for {updated} road segments")
    
    def update_edge(self, u: int, v: int, k: int, attributes: Dict) -> None:
        """
        Update attributes for a specific edge.
        
        Args:
            u: From node
            v: To node
            k: Edge key
            attributes: Dictionary of attributes to update
        """
        edge_key = (u, v, k)
        
        if edge_key in self.edges:
            self.edges[edge_key].update(attributes)
            
            # Update the graph as well
            for attr, value in attributes.items():
                self.graph[u][v][k][attr] = value
            
            logger.debug(f"Updated edge {edge_key} with {len(attributes)} attributes")
        else:
            logger.warning(f"Cannot update nonexistent edge {edge_key}")
    
    def add_blockage(self, lat: float, lon: float, radius: float, 
                    incident_type: str = "accident") -> List[Tuple[int, int, int]]:
        """
        Add a road blockage around a point.
        
        Args:
            lat: Latitude of incident
            lon: Longitude of incident
            radius: Radius in meters to block
            incident_type: Type of incident causing blockage
            
        Returns:
            List of edge keys that were blocked
        """
        # Find the nearest node
        incident_point = (lat, lon)
        nearest_node = None
        min_distance = float('inf')
        
        for node, coords in self.nodes.items():
            dist = self._haversine_distance(lat, lon, coords[0], coords[1])
            if dist < min_distance:
                min_distance = dist
                nearest_node = node
        
        if not nearest_node:
            logger.warning(f"Could not find nearest node to blockage at {lat}, {lon}")
            return []
            
        # Find all edges within the radius
        blocked_edges = []
        
        for edge_key, edge_data in self.edges.items():
            u, v, k = edge_key
            
            # Get node coordinates
            if u in self.nodes and v in self.nodes:
                u_coords = self.nodes[u]
                v_coords = self.nodes[v]
                
                # Check if either endpoint is within radius
                u_dist = self._haversine_distance(lat, lon, u_coords[0], u_coords[1])
                v_dist = self._haversine_distance(lat, lon, v_coords[0], v_coords[1])
                
                if u_dist <= radius or v_dist <= radius:
                    # Apply blockage based on incident type
                    if incident_type == "accident":
                        # Major slowdown but not completely blocked
                        new_attrs = {
                            'congestion': 0.9,
                            'travel_time': edge_data['travel_time'] * 5  # 5x slower
                        }
                    elif incident_type == "construction":
                        # Moderate slowdown
                        new_attrs = {
                            'congestion': 0.7,
                            'travel_time': edge_data['travel_time'] * 2  # 2x slower
                        }
                    elif incident_type == "road_closure":
                        # Complete blockage - set very high travel time
                        new_attrs = {
                            'congestion': 1.0,
                            'travel_time': edge_data['travel_time'] * 1000  # Effectively closed
                        }
                    else:
                        # Default moderate slowdown
                        new_attrs = {
                            'congestion': 0.5,
                            'travel_time': edge_data['travel_time'] * 1.5  # 1.5x slower
                        }
                    
                    # Update the edge
                    self.update_edge(u, v, k, new_attrs)
                    blocked_edges.append(edge_key)
        
        logger.info(f"Added blockage at {lat}, {lon} (radius: {radius}m): " +
                  f"{len(blocked_edges)} edges affected")
        
        return blocked_edges
    
    def clear_blockage(self, edge_keys: List[Tuple[int, int, int]]) -> None:
        """
        Clear a previously added blockage.
        
        Args:
            edge_keys: List of edge keys to clear
        """
        for edge_key in edge_keys:
            if edge_key in self.edges:
                u, v, k = edge_key
                
                # Reset to normal conditions
                length = self.edges[edge_key]['length']
                speed = self.edges[edge_key]['speed_limit']
                base_time = length / (speed / 3.6)
                
                new_attrs = {
                    'congestion': 0.0,
                    'travel_time': base_time
                }
                
                self.update_edge(u, v, k, new_attrs)
        
        logger.info(f"Cleared blockage affecting {len(edge_keys)} edges")
    
    def route(self, start_lat: float, start_lon: float, end_lat: float, end_lon: float,
             vehicle_type: str = "ambulance", avoid_edges: List = None) -> Tuple[List, float]:
        """
        Find the fastest route between two points.
        
        Args:
            start_lat: Start latitude
            start_lon: Start longitude
            end_lat: End latitude
            end_lon: End longitude
            vehicle_type: Type of vehicle for routing
            avoid_edges: List of edge keys to avoid
            
        Returns:
            Tuple of (route_points, estimated_time)
            where route_points is a list of (lat, lon) and time is in seconds
        """
        # Find nearest nodes to start and end points
        start_node = self._nearest_node(start_lat, start_lon)
        end_node = self._nearest_node(end_lat, end_lon)
        
        if start_node is None or end_node is None:
            logger.error(f"Cannot find valid start or end node for route")
            return [], 0
        
        # Set up edge weights based on vehicle type
        if vehicle_type == "ambulance":
            # Ambulances prioritize speed and can use special lanes
            weight_attr = 'travel_time'
            # Could modify weights here for ambulance-specific routing
        elif vehicle_type == "firetruck":
            # Firetrucks need wider roads and avoid tight turns
            weight_attr = 'travel_time'
            # Could modify weights here for firetruck-specific routing
        else:
            # Default routing
            weight_attr = 'travel_time'
        
        # Calculate route using modified Dijkstra algorithm
        try:
            # Use A* algorithm for faster routing
            route_nodes = nx.astar_path(
                self.graph, 
                start_node, 
                end_node, 
                weight=weight_attr,
                heuristic=self._a_star_heuristic
            )
            
            # Calculate total time
            total_time = 0
            route_points = []
            
            # Convert nodes to coordinates and calculate time
            for i in range(len(route_nodes) - 1):
                u, v = route_nodes[i], route_nodes[i+1]
                
                # Find the edge between these nodes (there might be multiple)
                min_time = float('inf')
                best_key = None
                
                for k in self.graph[u][v]:
                    edge_key = (u, v, k)
                    
                    # Skip edges to avoid
                    if avoid_edges and edge_key in avoid_edges:
                        continue
                        
                    time = self.graph[u][v][k].get(weight_attr, float('inf'))
                    if time < min_time:
                        min_time = time
                        best_key = k
                
                if best_key is not None:
                    total_time += min_time
                    
                    # Add midpoints if available for better visualization
                    if 'geometry' in self.graph[u][v][best_key]:
                        # Extract points from LineString geometry
                        geom = self.graph[u][v][best_key]['geometry']
                        for point in list(geom.coords):
                            route_points.append((point[1], point[0]))  # Swap to (lat, lon)
                    else:
                        # Just use endpoints
                        if u in self.nodes:
                            route_points.append(self.nodes[u])
            
            # Add the final node
            if route_nodes[-1] in self.nodes:
                route_points.append(self.nodes[route_nodes[-1]])
            
            # Adjust time for emergency vehicle
            # Emergency vehicles can typically travel faster than normal traffic
            if vehicle_type == "ambulance":
                # Ambulances are ~25% faster than normal traffic in emergency mode
                total_time *= 0.75
            elif vehicle_type == "firetruck":
                # Firetrucks are ~10% faster due to size/weight
                total_time *= 0.9
            elif vehicle_type == "police":
                # Police are ~30% faster
                total_time *= 0.7
            
            logger.info(f"Calculated route: {len(route_points)} points, " +
                      f"estimated time: {total_time:.1f}s")
            
            return route_points, total_time
            
        except Exception as e:
            logger.error(f"Error calculating route: {str(e)}")
            return [], 0
    
    def _nearest_node(self, lat: float, lon: float) -> int:
        """
        Find the nearest node to a point.
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            Nearest node ID
        """
        nearest_node = None
        min_distance = float('inf')
        
        for node, coords in self.nodes.items():
            node_lat, node_lon = coords
            dist = self._haversine_distance(lat, lon, node_lat, node_lon)
            
            if dist < min_distance:
                min_distance = dist
                nearest_node = node
        
        return nearest_node
    
    def _a_star_heuristic(self, u: int, v: int) -> float:
        """
        Heuristic function for A* algorithm.
        
        Args:
            u: Source node
            v: Target node
            
        Returns:
            Estimated cost (time) between nodes
        """
        if u in self.nodes and v in self.nodes:
            u_lat, u_lon = self.nodes[u]
            v_lat, v_lon = self.nodes[v]
            
            # Calculate straight-line distance in meters
            distance = self._haversine_distance(u_lat, u_lon, v_lat, v_lon)
            
            # Estimate time assuming direct route at maximum speed (80 km/h = 22.2 m/s)
            time_estimate = distance / 22.2
            
            return time_estimate
        
        return 0
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate the haversine distance between two points in meters.
        
        Args:
            lat1, lon1: First point coordinates
            lat2, lon2: Second point coordinates
            
        Returns:
            Distance in meters
        """
        # Radius of the Earth in meters
        R = 6371000
        
        # Convert to radians
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)
        
        # Differences
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        # Haversine formula
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c
    
    def get_traffic_pattern(self, edge_key: Tuple[int, int, int], 
                          hour: int = None) -> float:
        """
        Get historical traffic pattern for an edge at a specific hour.
        
        Args:
            edge_key: (from_node, to_node, key)
            hour: Hour of day (0-23), or None for current hour
            
        Returns:
            Average congestion level (0-1)
        """
        if hour is None:
            hour = datetime.now().hour
            
        if edge_key not in self.edge_history or not self.edge_history[edge_key]:
            return 0.0
            
        # Filter history entries for the specified hour
        hour_entries = []
        
        for timestamp, travel_time in self.edge_history[edge_key]:
            if timestamp.hour == hour:
                hour_entries.append(travel_time)
        
        if not hour_entries:
            return 0.0
            
        # Calculate average
        if edge_key in self.edges:
            base_time = self.edges[edge_key]['length'] / (self.edges[edge_key]['speed_limit'] / 3.6)
            avg_time = sum(hour_entries) / len(hour_entries)
            
            # Calculate congestion factor
            if base_time > 0:
                congestion = min(1.0, max(0.0, (avg_time / base_time - 1) / DEFAULT_CONGESTION_WEIGHT))
                return congestion
        
        return 0.0
    
    def visualize_network(self, route: List[Tuple[float, float]] = None, 
                        incidents: List[Tuple[float, float, str]] = None) -> str:
        """
        Create a visualization of the road network with optional route and incidents.
        
        Args:
            route: Optional list of (lat, lon) points for a route
            incidents: Optional list of (lat, lon, type) for incidents
            
        Returns:
            Folium map HTML as string
        """
        # Create a folium map centered at the average of node coordinates
        lats = [coords[0] for coords in self.nodes.values()]
        lons = [coords[1] for coords in self.nodes.values()]
        
        if not lats or not lons:
            # Default center if no nodes
            center = [37.7749, -122.4194]  # San Francisco
        else:
            center = [sum(lats) / len(lats), sum(lons) / len(lons)]
        
        m = folium.Map(location=center, zoom_start=13)
        
        # Add road network edges colored by congestion
        added_edges = set()
        
        for edge_key, edge_data in self.edges.items():
            u, v, k = edge_key
            
            if (u, v) in added_edges:
                continue  # Skip parallel edges for visualization
                
            if u in self.nodes and v in self.nodes:
                u_lat, u_lon = self.nodes[u]
                v_lat, v_lon = self.nodes[v]
                
                # Color based on congestion
                congestion = edge_data.get('congestion', 0)
                
                if congestion < 0.3:
                    color = 'green'
                elif congestion < 0.6:
                    color = 'orange'
                else:
                    color = 'red'
                
                # Create popup with edge info
                popup_text = f"""
                Road Type: {edge_data.get('road_type', 'Unknown')}<br>
                Speed Limit: {edge_data.get('speed_limit', 0)} km/h<br>
                Length: {edge_data.get('length', 0):.1f} m<br>
                Congestion: {congestion:.2f}<br>
                Travel Time: {edge_data.get('travel_time', 0):.1f} s
                """
                
                # Add the edge to the map
                folium.PolyLine(
                    locations=[(u_lat, u_lon), (v_lat, v_lon)],
                    color=color,
                    weight=2,
                    opacity=0.7,
                    popup=popup_text
                ).add_to(m)
                
                added_edges.add((u, v))
        
        # Add the route if provided
        if route:
            folium.PolyLine(
                locations=route,
                color='blue',
                weight=5,
                opacity=0.8,
                popup='Emergency Route'
            ).add_to(m)
        
        # Add incidents if provided
        if incidents:
            for lat, lon, incident_type in incidents:
                # Icon and color based on incident type
                if incident_type in ['accident', 'crash']:
                    icon = folium.Icon(color='red', icon='car', prefix='fa')
                elif incident_type in ['fire']:
                    icon = folium.Icon(color='red', icon='fire', prefix='fa')
                elif incident_type in ['medical']:
                    icon = folium.Icon(color='red', icon='plus', prefix='fa')
                else:
                    icon = folium.Icon(color='red', icon='exclamation', prefix='fa')
                
                folium.Marker(
                    location=[lat, lon],
                    popup=f"Incident: {incident_type}",
                    icon=icon
                ).add_to(m)
        
        # Convert to HTML string
        return m._repr_html_()
    
    def save_to_file(self, filepath: str) -> None:
        """
        Save the road network to a file.
        
        Args:
            filepath: Path to save file
        """
        # Create a dictionary with all necessary data
        data = {
            'city_name': self.city_name,
            'bbox': self.bbox,
            'last_traffic_update': self.last_traffic_update,
            'nodes': self.nodes,
            'edges': self.edges
        }
        
        # Save network data (not the graph itself)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
            
        logger.info(f"Saved road network to {filepath}")
    
    def load_from_file(self, filepath: str) -> bool:
        """
        Load the road network from a file.
        
        Args:
            filepath: Path to load file from
            
        Returns:
            True if loaded successfully
        """
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            # Update attributes
            self.city_name = data['city_name']
            self.bbox = data['bbox']
            self.last_traffic_update = data['last_traffic_update']
            self.nodes = data['nodes']
            self.edges = data['edges']
            
            # Recreate the graph
            self.graph = nx.MultiDiGraph()
            
            # Add nodes
            for node_id, (lat, lon) in self.nodes.items():
                self.graph.add_node(node_id, y=lat, x=lon)
            
            # Add edges
            for (u, v, k), attrs in self.edges.items():
                self.graph.add_edge(u, v, key=k, **attrs)
            
            logger.info(f"Loaded road network from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading road network: {str(e)}")
            return False


class EmergencyRouter:
    """
    Main class for emergency response routing.
    
    Manages vehicles, stations, incidents, and routing decisions.
    """
    
    def __init__(self, city_name: str = None, road_network_file: str = None):
        """
        Initialize the emergency router.
        
        Args:
            city_name: Name of the city for loading OSM data
            road_network_file: Optional path to pre-saved road network
        """
        # Initialize collections
        self.vehicles = {}  # Dict of vehicle_id -> EmergencyVehicle
        self.stations = {}  # Dict of station_id -> EmergencyStation
        self.incidents = {}  # Dict of incident_id -> incident info
        self.active_blockages = {}  # Dict of incident_id -> list of blocked edges
        
        # Create or load road network
        if road_network_file and os.path.exists(road_network_file):
            self.road_network = RoadNetwork()
            self.road_network.load_from_file(road_network_file)
        else:
            self.road_network = RoadNetwork(city_name=city_name)
            
            # Save the network if it was created successfully
            if self.road_network.graph and self.road_network.graph.number_of_nodes() > 0:
                os.makedirs("data", exist_ok=True)
                self.road_network.save_to_file(f"data/{city_name or 'custom'}_network.pkl")
        
        # External API connectors
        self.traffic_connector = None
        self.dispatch_connector = None
        
        # Metrics tracking
        self.response_times = []
        self.dispatches_by_type = defaultdict(int)
        self.coverage_gaps = []
        
        logger.info(f"Initialized emergency router for {city_name or 'custom area'}")
    
    def add_vehicle(self, vehicle: EmergencyVehicle) -> None:
        """
        Add a vehicle to the system.
        
        Args:
            vehicle: EmergencyVehicle to add
        """
        self.vehicles[vehicle.id] = vehicle
        
        # Add to station if applicable
        if vehicle.station_id and vehicle.station_id in self.stations:
            self.stations[vehicle.station_id].add_vehicle(vehicle.id)
            
        logger.debug(f"Added vehicle {vehicle.id} to the system")
    
    def add_station(self, station: EmergencyStation) -> None:
        """
        Add a station to the system.
        
        Args:
            station: EmergencyStation to add
        """
        self.stations[station.id] = station
        logger.debug(f"Added station {station.id} to the system")
    
    def register_incident(self, incident_id: str, lat: float, lon: float, 
                        incident_type: str, priority: int, 
                        resources_needed: Dict[str, int] = None) -> None:
        """
        Register a new incident that requires emergency response.
        
        Args:
            incident_id: Unique incident identifier
            lat, lon: Location coordinates
            incident_type: Type of incident (medical, fire, police, etc.)
            priority: Priority level (1-5, 1 highest)
            resources_needed: Dict of {vehicle_type: count} needed
        """
        # Create incident record
        timestamp = datetime.now()
        resources_needed = resources_needed or {}
        
        incident = {
            'id': incident_id,
            'latitude': lat,
            'longitude': lon,
            'type': incident_type,
            'priority': priority,
            'resources_needed': resources_needed,
            'resources_assigned': {},
            'timestamp': timestamp,
            'status': 'active',
            'eta': None
        }
        
        self.incidents[incident_id] = incident
        
        # Add traffic blockage if this incident affects roads
        affects_roads = incident_type in ['accident', 'fire', 'flood', 'road_closure']
        
        if affects_roads:
            radius_map = {
                'accident': 100,  # 100m radius
                'fire': 200,      # 200m radius
                'flood': 300,     # 300m radius
                'road_closure': 150  # 150m radius
            }
            
            radius = radius_map.get(incident_type, 100)
            blocked_edges = self.road_network.add_blockage(lat, lon, radius, incident_type)
            
            if blocked_edges:
                self.active_blockages[incident_id] = blocked_edges
        
        logger.info(f"Registered incident {incident_id} of type {incident_type} " +
                  f"at {lat},{lon} (priority {priority})")
    
    def close_incident(self, incident_id: str) -> bool:
        """
        Mark an incident as closed and release resources.
        
        Args:
            incident_id: ID of incident to close
            
        Returns:
            True if closed successfully
        """
        if incident_id not in self.incidents:
            logger.warning(f"Cannot close unknown incident {incident_id}")
            return False
            
        # Update incident status
        self.incidents[incident_id]['status'] = 'closed'
        self.incidents[incident_id]['close_time'] = datetime.now()
        
        # Release any assigned vehicles
        assigned_vehicles = self.incidents[incident_id]['resources_assigned']
        
        for vehicle_id in assigned_vehicles.get('vehicle_ids', []):
            if vehicle_id in self.vehicles:
                vehicle = self.vehicles[vehicle_id]
                
                # Only update if still assigned to this incident
                if vehicle.current_incident_id == incident_id:
                    # Set status to returning
                    vehicle.update_status('returning')
                    
                    # Route back to station
                    if vehicle.location and vehicle.station_id in self.stations:
                        station = self.stations[vehicle.station_id]
                        
                        route, time = self.road_network.route(
                            vehicle.location[0], vehicle.location[1],
                            station.location[0], station.location[1],
                            vehicle.type
                        )
                        
                        vehicle.set_route(route, time)
                        
                        logger.debug(f"Vehicle {vehicle_id} returning to station {station.id}, " +
                                    f"ETA: {time:.1f}s")
        
        # Clear any traffic blockages
        if incident_id in self.active_blockages:
            self.road_network.clear_blockage(self.active_blockages[incident_id])
            del self.active_blockages[incident_id]
        
        logger.info(f"Closed incident {incident_id}")
        return True
    
    def dispatch_vehicles(self, incident_id: str) -> Dict:
        """
        Find and dispatch the best available vehicles for an incident.
        
        Args:
            incident_id: ID of incident to dispatch for
            
        Returns:
            Dictionary with dispatch results
        """
        if incident_id not in self.incidents:
            logger.warning(f"Cannot dispatch for unknown incident {incident_id}")
            return {'success': False, 'message': 'Incident not found'}
            
        incident = self.incidents[incident_id]
        
        # Skip if already closed
        if incident['status'] == 'closed':
            return {'success': False, 'message': 'Incident already closed'}
            
        # Get incident details
        lat, lon = incident['latitude'], incident['longitude']
        incident_type = incident['type']
        priority = incident['priority']
        resources_needed = incident['resources_needed']
        
        # Determine vehicle types needed if not specified
        if not resources_needed:
            if incident_type in ['medical', 'heart_attack', 'injury']:
                resources_needed = {'ambulance': 1}
            elif incident_type in ['fire', 'gas_leak', 'explosion']:
                resources_needed = {'firetruck': 1, 'ambulance': 1}
            elif incident_type in ['accident', 'crash']:
                resources_needed = {'ambulance': 1, 'police': 1}
            elif incident_type in ['crime', 'violence']:
                resources_needed = {'police': 2}
            else:
                # Default for unknown types
                resources_needed = {'police': 1}
        
        # Track resources that still need to be assigned
        resources_still_needed = resources_needed.copy()
        
        # Track vehicles assigned in this dispatch
        newly_assigned = {'vehicle_ids': []}
        
        # Find available vehicles of each type, sorted by estimated arrival time
        for vehicle_type, count_needed in resources_needed.items():
            count_to_assign = count_needed - sum(1 for v in self.vehicles.values() 
                                            if v.current_incident_id == incident_id and 
                                            v.type == vehicle_type)
            
            if count_to_assign <= 0:
                # Already have enough of this type
                resources_still_needed[vehicle_type] = 0
                continue
                
            # Find available vehicles of this type
            available_vehicles = [v for v in self.vehicles.values() 
                                if v.type == vehicle_type and 
                                v.status == 'available']
            
            if not available_vehicles:
                logger.warning(f"No available {vehicle_type} vehicles for incident {incident_id}")
                continue
                
            # Calculate ETA for each vehicle
            vehicles_with_eta = []
            
            for vehicle in available_vehicles:
                # Get vehicle location (either current location or station)
                if vehicle.location:
                    start_lat, start_lon = vehicle.location
                elif vehicle.station_id in self.stations:
                    station = self.stations[vehicle.station_id]
                    start_lat, start_lon = station.location
                else:
                    continue  # Skip if no valid location
                
                # Calculate route and ETA
                route, eta = self.road_network.route(
                    start_lat, start_lon, lat, lon, vehicle.type
                )
                
                if route and eta > 0:
                    vehicles_with_eta.append((vehicle, route, eta))
            
            # Sort by ETA (fastest first)
            vehicles_with_eta.sort(key=lambda x: x[2])
            
            # Assign vehicles up to the count needed
            for i in range(min(count_to_assign, len(vehicles_with_eta))):
                vehicle, route, eta = vehicles_with_eta[i]
                
                # Dispatch the vehicle
                vehicle.update_status('dispatched', incident_id)
                vehicle.set_route(route, eta)
                
                # Update incident record
                if 'vehicle_ids' not in incident['resources_assigned']:
                    incident['resources_assigned']['vehicle_ids'] = []
                
                incident['resources_assigned']['vehicle_ids'].append(vehicle.id)
                newly_assigned['vehicle_ids'].append(vehicle.id)
                
                # Update eta for incident (use fastest vehicle's ETA)
                if i == 0:
                    incident['eta'] = eta
                
                # Decrement count still needed
                resources_still_needed[vehicle_type] -= 1
                
                logger.info(f"Dispatched {vehicle_type} {vehicle.id} to incident {incident_id}, " +
                          f"ETA: {eta:.1f}s")
        
        # Check if we assigned all needed resources
        all_assigned = all(count <= 0 for count in resources_still_needed.values())
        
        if all_assigned:
            logger.info(f"Successfully assigned all needed resources to incident {incident_id}")
        else:
            # Log which resources are still needed
            for vehicle_type, count in resources_still_needed.items():
                if count > 0:
                    logger.warning(f"Still need {count} {vehicle_type} for incident {incident_id}")
        
        # Return dispatch results
        return {
            'success': True,
            'incident_id': incident_id,
            'all_resources_assigned': all_assigned,
            'vehicles_assigned': newly_assigned['vehicle_ids'],
            'resources_still_needed': {k: v for k, v in resources_still_needed.items() if v > 0},
            'eta': incident['eta']
        }
    
    def update_vehicle_locations(self, location_updates: List[Dict]) -> None:
        """
        Update vehicle locations from GPS or other tracking data.
        
        Args:
            location_updates: List of {vehicle_id, lat, lon, timestamp} dictionaries
        """
        for update in location_updates:
            vehicle_id = update.get('vehicle_id')
            lat = update.get('latitude')
            lon = update.get('longitude')
            timestamp = update.get('timestamp')
            
            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp)
                except ValueError:
                    timestamp = datetime.now()
            
            if not vehicle_id or vehicle_id not in self.vehicles:
                logger.warning(f"Location update for unknown vehicle {vehicle_id}")
                continue
                
            if lat is None or lon is None:
                logger.warning(f"Invalid location update for vehicle {vehicle_id}")
                continue
                
            # Update the vehicle location
            self.vehicles[vehicle_id].update_location(lat, lon, timestamp)
            
            # Check if vehicle has arrived at incident
            vehicle = self.vehicles[vehicle_id]
            
            if (vehicle.status == 'dispatched' and 
                vehicle.current_incident_id and 
                vehicle.current_incident_id in self.incidents):
                
                incident = self.incidents[vehicle.current_incident_id]
                
                # Calculate distance to incident
                distance = self.road_network._haversine_distance(
                    lat, lon, incident['latitude'], incident['longitude']
                )
                
                # Consider arrived if within 50 meters
                if distance <= 50:
                    logger.info(f"Vehicle {vehicle_id} arrived at incident {vehicle.current_incident_id}")
                    
                    # Calculate response time
                    if 'timestamp' in incident:
                        response_time = (datetime.now() - incident['timestamp']).total_seconds()
                        self.response_times.append(response_time)
                        
                        logger.info(f"Response time for incident {vehicle.current_incident_id}: " +
                                  f"{response_time:.1f}s")
                        
                        # Record in station statistics
                        if vehicle.station_id in self.stations:
                            self.stations[vehicle.station_id].record_dispatch(
                                vehicle.type, response_time, incident['type']
                            )
                    
                    # Update dispatch counts
                    self.dispatches_by_type[incident['type']] += 1
            
            # Check if vehicle has returned to station
            elif vehicle.status == 'returning' and vehicle.station_id in self.stations:
                station = self.stations[vehicle.station_id]
                
                # Calculate distance to station
                distance = self.road_network._haversine_distance(
                    lat, lon, station.location[0], station.location[1]
                )
                
                # Consider arrived if within 50 meters
                if distance <= 50:
                    logger.info(f"Vehicle {vehicle_id} returned to station {vehicle.station_id}")
                    vehicle.update_status('available')
    
    def update_traffic_data(self) -> None:
        """Update traffic data from external sources."""
        # Skip if no traffic connector
        if not self.traffic_connector:
            return
            
        # Check if it's time to update
        if (self.road_network.last_traffic_update and 
            (datetime.now() - self.road_network.last_traffic_update).total_seconds() < TRAFFIC_UPDATE_INTERVAL):
            return
            
        try:
            # Get traffic data from connector
            traffic_data = self.traffic_connector.get_traffic_data()
            
            if traffic_data:
                self.road_network.update_traffic(traffic_data)
        except Exception as e:
            logger.error(f"Error updating traffic data: {str(e)}")
    
    def analyze_coverage(self) -> Dict:
        """
        Analyze emergency service coverage across the city.
        
        Returns:
            Coverage statistics
        """
        # Can't analyze if we don't have stations
        if not self.stations:
            return {
                'coverage_percentage': 0,
                'uncovered_areas': [],
                'average_response_time': 0
            }
            
        # Get all node coordinates in the road network
        node_coords = list(self.nodes.values())
        
        # Calculate coverage for each node
        covered_nodes = 0
        response_times = []
        uncovered_areas = []
        
        for lat, lon in node_coords:
            # Find nearest station of each type
            nearest_stations = {
                'fire': (None, float('inf')),
                'hospital': (None, float('inf')),
                'police': (None, float('inf'))
            }
            
            for station_id, station in self.stations.items():
                station_type = 'fire' if station.type == 'fire' else \
                              'hospital' if station.type == 'hospital' else \
                              'police' if station.type == 'police' else \
                              station.type  # Fallback
                
                # Skip if not a recognized type
                if station_type not in nearest_stations:
                    continue
                    
                # Calculate distance
                dist = self.road_network._haversine_distance(
                    lat, lon, station.location[0], station.location[1]
                )
                
                # Update if closer
                if dist < nearest_stations[station_type][1]:
                    nearest_stations[station_type] = (station_id, dist)
            
            # Check if node is covered by at least one service within reasonable distance
            max_distances = {
                'fire': 5000,    # 5km for fire
                'hospital': 10000,  # 10km for ambulance
                'police': 8000    # 8km for police
            }
            
            is_covered = False
            for service_type, (station_id, dist) in nearest_stations.items():
                if station_id and dist <= max_distances[service_type]:
                    is_covered = True
                    
                    # Estimate response time based on distance
                    speed_kmh = 40  # Assume 40 km/h average speed
                    time_minutes = (dist / 1000) / (speed_kmh / 60)
                    response_times.append(time_minutes)
                    
                    break
            
            if is_covered:
                covered_nodes += 1
            else:
                # This is an uncovered area
                uncovered_areas.append((lat, lon))
        
        # Calculate coverage percentage
        coverage_pct = (covered_nodes / len(node_coords)) * 100 if node_coords else 0
        
        # Calculate average response time
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        # Store uncovered areas for future reference
        self.coverage_gaps = uncovered_areas
        
        return {
            'coverage_percentage': coverage_pct,
            'uncovered_areas': uncovered_areas,
            'average_response_time': avg_response_time,
            'service_distribution': {
                'fire_stations': len([s for s in self.stations.values() if s.type == 'fire']),
                'hospitals': len([s for s in self.stations.values() if s.type == 'hospital']),
                'police_stations': len([s for s in self.stations.values() if s.type == 'police'])
            }
        }
    
    def recommend_new_station(self) -> Dict:
        """
        Recommend location for a new emergency station.
        
        Returns:
            Recommendation details
        """
        # Need coverage gaps to make a recommendation
        if not self.coverage_gaps:
            self.analyze_coverage()
            
        if not self.coverage_gaps:
            return {
                'success': False,
                'message': 'No coverage gaps found'
            }
            
        # Use clustering to find the best location for a new station
        try:
            from sklearn.cluster import KMeans
            
            # Convert coverage gaps to numpy array
            points = np.array(self.coverage_gaps)
            
            # Use k-means to find the center of the largest cluster
            kmeans = KMeans(n_clusters=min(5, len(points)), random_state=42)
            clusters = kmeans.fit_predict(points)
            
            # Find the cluster with the most points
            cluster_sizes = [sum(clusters == i) for i in range(kmeans.n_clusters)]
            largest_cluster = np.argmax(cluster_sizes)
            
            # Get the center of the largest cluster
            new_location = kmeans.cluster_centers_[largest_cluster]
            
            # Calculate potential coverage improvement
            potential_coverage = (len([p for i, p in enumerate(self.coverage_gaps) 
                                    if clusters[i] == largest_cluster]) / 
                                len(self.coverage_gaps) * 100)
            
            # Determine station type based on current distribution
            station_counts = {
                'fire': len([s for s in self.stations.values() if s.type == 'fire']),
                'hospital': len([s for s in self.stations.values() if s.type == 'hospital']),
                'police': len([s for s in self.stations.values() if s.type == 'police'])
            }
            
            needed_type = min(station_counts, key=station_counts.get)
            
            return {
                'success': True,
                'latitude': float(new_location[0]),
                'longitude': float(new_location[1]),
                'recommended_type': needed_type,
                'potential_coverage_improvement': potential_coverage,
                'current_coverage_percentage': self.analyze_coverage()['coverage_percentage']
            }
            
        except Exception as e:
            logger.error(f"Error generating station recommendation: {str(e)}")
            return {
                'success': False,
                'message': f"Error: {str(e)}"
            }
    
    def get_statistics(self) -> Dict:
        """
        Get system-wide statistics.
        
        Returns:
            Dictionary of statistics
        """
        # Calculate statistics
        stats = {
            'vehicles': {
                'total': len(self.vehicles),
                'available': sum(1 for v in self.vehicles.values() if v.status == 'available'),
                'dispatched': sum(1 for v in self.vehicles.values() if v.status == 'dispatched'),
                'by_type': {
                    'ambulance': sum(1 for v in self.vehicles.values() if v.type == 'ambulance'),
                    'firetruck': sum(1 for v in self.vehicles.values() if v.type == 'firetruck'),
                    'police': sum(1 for v in self.vehicles.values() if v.type == 'police')
                }
            },
            'stations': {
                'total': len(self.stations),
                'by_type': {
                    'fire': sum(1 for s in self.stations.values() if s.type == 'fire'),
                    'hospital': sum(1 for s in self.stations.values() if s.type == 'hospital'),
                    'police': sum(1 for s in self.stations.values() if s.type == 'police'),
                    'combined': sum(1 for s in self.stations.values() if s.type == 'combined')
                }
            },
            'incidents': {
                'total': len(self.incidents),
                'active': sum(1 for i in self.incidents.values() if i['status'] == 'active'),
                'closed': sum(1 for i in self.incidents.values() if i['status'] == 'closed'),
                'by_type': dict(self.dispatches_by_type)
            },
            'response_times': {
                'average': sum(self.response_times) / len(self.response_times) if self.response_times else 0,
                'min': min(self.response_times) if self.response_times else 0,
                'max': max(self.response_times) if self.response_times else 0
            },
            'coverage': self.analyze_coverage()
        }
        
        return stats
    
    def save_to_file(self, filepath: str) -> None:
        """
        Save the complete emergency routing system to a file.
        
        Args:
            filepath: Path to save file
        """
        # Create a dictionary with all data
        data = {
            'vehicles': {v_id: v.to_dict() for v_id, v in self.vehicles.items()},
            'stations': {s_id: s.to_dict() for s_id, s in self.stations.items()},
            'incidents': self.incidents,
            'response_times': self.response_times,
            'dispatches_by_type': dict(self.dispatches_by_type),
            'coverage_gaps': self.coverage_gaps
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # Save to file
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
            
        logger.info(f"Saved emergency routing system to {filepath}")
    
    def load_from_file(self, filepath: str) -> bool:
        """
        Load the emergency routing system from a file.
        
        Args:
            filepath: Path to load file from
            
        Returns:
            True if loaded successfully
        """
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            # Recreate vehicles
            for v_id, v_data in data['vehicles'].items():
                vehicle = EmergencyVehicle(
                    v_data['id'],
                    v_data['type'],
                    v_data['station_id'],
                    v_data.get('capabilities', []),
                    v_data.get('location')
                )
                vehicle.status = v_data['status']
                vehicle.current_incident_id = v_data.get('current_incident_id')
                vehicle.route_time_estimate = v_data.get('route_time_estimate', 0)
                
                self.vehicles[v_id] = vehicle
            
            # Recreate stations
            for s_id, s_data in data['stations'].items():
                station = EmergencyStation(
                    s_data['id'],
                    s_data['type'],
                    s_data['location'],
                    s_data.get('address'),
                    s_data.get('capacity', 5)
                )
                station.vehicles = s_data.get('vehicles', [])
                station.total_dispatches = s_data.get('total_dispatches', 0)
                station.avg_response_time = s_data.get('avg_response_time', 0)
                station.dispatches_by_hour = s_data.get('dispatches_by_hour', [0] * 24)
                
                if 'dispatches_by_type' in s_data:
                    station.dispatches_by_type = defaultdict(int)
                    for k, v in s_data['dispatches_by_type'].items():
                        station.dispatches_by_type[k] = v
                
                self.stations[s_id] = station
            
            # Load other data
            self.incidents = data['incidents']
            self.response_times = data['response_times']
            
            self.dispatches_by_type = defaultdict(int)
            for k, v in data.get('dispatches_by_type', {}).items():
                self.dispatches_by_type[k] = v
                
            self.coverage_gaps = data.get('coverage_gaps', [])
            
            logger.info(f"Loaded emergency routing system from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading emergency routing system: {str(e)}")
            return False


class TrafficDataConnector:
    """
    Connector for real-time traffic data APIs.
    
    This class handles communication with external traffic data sources
    and converts the data to the format needed by the routing system.
    """
    
    def __init__(self, api_key: str = None, api_url: str = None):
        """
        Initialize the traffic data connector.
        
        Args:
            api_key: API key for the traffic data service
            api_url: URL of the traffic data API
        """
        self.api_key = api_key
        self.api_url = api_url
        self.last_update = None
        self.cache_expiry = timedelta(minutes=5)
        self.cached_data = None
        
        logger.debug(f"Initialized traffic data connector")
    
    def get_traffic_data(self) -> Dict:
        """
        Get current traffic data.
        
        Returns:
            Dictionary mapping edge IDs to congestion values
        """
        # Check if cache is still valid
        if (self.cached_data and self.last_update and 
            datetime.now() - self.last_update < self.cache_expiry):
            return self.cached_data
            
        # If no API details, use simulated data
        if not self.api_key or not self.api_url:
            return self._generate_simulated_data()
            
        # Try to get data from API
        try:
            # Try multiple times in case of transient failures
            for attempt in range(API_RETRY_ATTEMPTS):
                # Send request to traffic API
                response = requests.get(
                    self.api_url,
                    params={'key': self.api_key, 'format': 'json'},
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Process the data into the expected format
                    # Format will depend on the specific API used
                    traffic_data = {}
                    
                    # Example processing for a hypothetical API
                    for record in data.get('traffic', []):
                        if 'segment_id' in record and 'congestion' in record:
                            # Map API segment ID to our edge ID format
                            edge_id = record['segment_id']  # This mapping would be API-specific
                            congestion = float(record['congestion']) / 100.0  # Normalize to 0-1
                            
                            traffic_data[edge_id] = congestion
                    
                    # Update cache
                    self.cached_data = traffic_data
                    self.last_update = datetime.now()
                    
                    logger.info(f"Updated traffic data: {len(traffic_data)} segments")
                    return traffic_data
                
                elif response.status_code == 429:  # Rate limit
                    # Wait and retry
                    logger.warning(f"Traffic API rate limit hit, retrying in 3 seconds")
                    time.sleep(3)
                else:
                    # Other error
                    logger.warning(f"Traffic API error (HTTP {response.status_code}): {response.text}")
                    break
            
            # If we reach here, all attempts failed
            logger.error(f"Failed to get traffic data after {API_RETRY_ATTEMPTS} attempts")
            
            # Fall back to simulated data
            return self._generate_simulated_data()
            
        except Exception as e:
            logger.error(f"Error getting traffic data: {str(e)}")
            return self._generate_simulated_data()
    
    def _generate_simulated_data(self) -> Dict:
        """
        Generate simulated traffic data for testing.
        
        Returns:
            Dictionary mapping edge IDs to congestion values
        """
        # Create random traffic data based on time of day
        hour = datetime.now().hour
        
        # Define base congestion levels by hour
        if 7 <= hour <= 9 or 16 <= hour <= 18:  # Rush hours
            base_congestion = 0.7
        elif 10 <= hour <= 15:  # Daytime
            base_congestion = 0.4
        elif 19 <= hour <= 22:  # Evening
            base_congestion = 0.3
        else:  # Night
            base_congestion = 0.1
            
        # Generate random congestion values for a set of edges
        traffic_data = {}
        
        # Simulate ~10000 road segments
        for i in range(10000):
            # Create edge ID in the format expected by the router
            u = random.randint(1, 1000)
            v = random.randint(1, 1000)
            k = 0
            edge_id = f"{u},{v},{k}"
            
            # Add random variation to base congestion
            variation = random.uniform(-0.2, 0.2)
            congestion = max(0, min(1, base_congestion + variation))
            
            traffic_data[edge_id] = congestion
        
        # Update cache
        self.cached_data = traffic_data
        self.last_update = datetime.now()
        
        logger.debug(f"Generated simulated traffic data for {len(traffic_data)} edges")
        return traffic_data


def load_sample_data(router: EmergencyRouter) -> None:
    """
    Load sample data for testing.
    
    Args:
        router: EmergencyRouter to populate
    """
    # Create some police stations
    police_stations = [
        EmergencyStation("police_1", "police", (37.775, -122.417), "123 Main St", 5),
        EmergencyStation("police_2", "police", (37.792, -122.400), "456 Market St", 3),
        EmergencyStation("police_3", "police", (37.758, -122.435), "789 Folsom St", 4)
    ]
    
    # Create some fire stations
    fire_stations = [
        EmergencyStation("fire_1", "fire", (37.785, -122.430), "100 Fire Rd", 4),
        EmergencyStation("fire_2", "fire", (37.765, -122.410), "200 Hydrant Ave", 5)
    ]
    
    # Create some hospitals
    hospitals = [
        EmergencyStation("hospital_1", "hospital", (37.780, -122.420), "SF General Hospital", 8),
        EmergencyStation("hospital_2", "hospital", (37.795, -122.440), "UCSF Medical Center", 10)
    ]
    
    # Add all stations
    for station in police_stations + fire_stations + hospitals:
        router.add_station(station)
    
    # Create some police vehicles
    police_vehicles = [
        EmergencyVehicle("police_1_1", "police", "police_1", ["K9"]),
        EmergencyVehicle("police_1_2", "police", "police_1"),
        EmergencyVehicle("police_2_1", "police", "police_2"),
        EmergencyVehicle("police_3_1", "police", "police_3", ["SWAT"]),
        EmergencyVehicle("police_3_2", "police", "police_3")
    ]
    
    # Create some fire vehicles
    fire_vehicles = [
        EmergencyVehicle("fire_1_1", "firetruck", "fire_1", ["ladder"]),
        EmergencyVehicle("fire_1_2", "firetruck", "fire_1"),
        EmergencyVehicle("fire_2_1", "firetruck", "fire_2", ["hazmat"]),
        EmergencyVehicle("fire_2_2", "firetruck", "fire_2")
    ]
    
    # Create some ambulances
    ambulances = [
        EmergencyVehicle("ambulance_1_1", "ambulance", "hospital_1"),
        EmergencyVehicle("ambulance_1_2", "ambulance", "hospital_1"),
        EmergencyVehicle("ambulance_1_3", "ambulance", "hospital_1"),
        EmergencyVehicle("ambulance_2_1", "ambulance", "hospital_2"),
        EmergencyVehicle("ambulance_2_2", "ambulance", "hospital_2", ["advanced_life_support"])
    ]
    
    # Add all vehicles
    for vehicle in police_vehicles + fire_vehicles + ambulances:
        router.add_vehicle(vehicle)
    
    # Create some incidents
    router.register_incident(
        "incident_1", 
        37.77, -122.42, 
        "accident", 
        2, 
        {"ambulance": 1, "police": 1}
    )
    
    router.register_incident(
        "incident_2", 
        37.79, -122.43, 
        "fire", 
        1, 
        {"firetruck": 2, "ambulance": 1}
    )
    
    router.register_incident(
        "incident_3", 
        37.76, -122.41, 
        "medical", 
        3, 
        {"ambulance": 1}
    )
    
    logger.info("Loaded sample data into emergency router")


def run_emergency_routing_system(city_name: str = "San Francisco", load_sample: bool = True):
    """
    Run the emergency routing system.
    
    Args:
        city_name: Name of the city to use
        load_sample: Whether to load sample data
    """
    logger.info(f"Starting emergency routing system for {city_name}")
    
    # Initialize the router
    router = EmergencyRouter(city_name=city_name)
    
    # Load sample data if requested
    if load_sample:
        load_sample_data(router)
    
    # Set up traffic connector
    router.traffic_connector = TrafficDataConnector()
    
    # Set up metrics and coverage analysis
    coverage = router.analyze_coverage()
    logger.info(f"Initial coverage analysis: {coverage['coverage_percentage']:.1f}% covered")
    
    # Dispatch to incidents
    for incident_id in router.incidents:
        result = router.dispatch_vehicles(incident_id)
        logger.info(f"Dispatch result for {incident_id}: {result['success']}")
    
    # Save the system state
    os.makedirs("data", exist_ok=True)
    router.save_to_file(f"data/{city_name.replace(' ', '_').lower()}_emergency_system.pkl")
    
    # Print final statistics
    stats = router.get_statistics()
    logger.info(f"System statistics: {stats['vehicles']['total']} vehicles, " +
              f"{stats['stations']['total']} stations, {stats['incidents']['total']} incidents")
    
    # Recommend a new station location if coverage is below 90%
    if coverage['coverage_percentage'] < 90:
        recommendation = router.recommend_new_station()
        if recommendation['success']:
            logger.info(f"Recommended new {recommendation['recommended_type']} station at " +
                      f"{recommendation['latitude']}, {recommendation['longitude']}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Emergency Response Routing System')
    parser.add_argument('--city', type=str, default="San Francisco", help='City name')
    parser.add_argument('--load_sample', action='store_true', help='Load sample data')
    parser.add_argument('--load_file', type=str, help='Load system from file')
    parser.add_argument('--save_file', type=str, help='Save system to file')
    
    args = parser.parse_args()
    
    if args.load_file:
        # Load from file
        router = EmergencyRouter()
        if router.load_from_file(args.load_file):
            logger.info(f"Loaded system from {args.load_file}")
            
            # Run some dispatches for testing
            for incident_id in router.incidents:
                if router.incidents[incident_id]['status'] == 'active':
                    router.dispatch_vehicles(incident_id)
            
            # Save if requested
            if args.save_file:
                router.save_to_file(args.save_file)
        else:
            logger.error(f"Failed to load from {args.load_file}")
    else:
        # Create new system
        run_emergency_routing_system(args.city, args.load_sample)

"""
SUMMARY:
========
This module implements an emergency response routing system that optimizes routes 
for emergency vehicles, taking into account real-time traffic conditions, road 
blockages, and historical response patterns. The system manages vehicles, stations, 
and incidents while finding the fastest routes to emergency locations.

Key components:
1. EmergencyVehicle - Represents emergency vehicles with routing capabilities
2. EmergencyStation - Manages stations and their assigned vehicles
3. RoadNetwork - Maintains the road graph with traffic conditions
4. EmergencyRouter - Core system that dispatches vehicles and analyzes coverage

The system can simulate traffic patterns, recommend optimal station locations,
and provide visualizations of emergency response networks.

TODO:
=====
1. Implement multi-vehicle coordination for complex incidents
2. Add predictive traffic modeling to anticipate congestion
3. Create real-time dashboard for emergency managers
4. Implement automatic rerouting when road conditions change
5. Add integration with traffic light preemption systems
6. Improve ETA calculation with machine learning on historical data
7. Build a web-based visualization tool for coverage analysis
8. Add support for drone-based emergency response units
9. Implement cross-jurisdiction coordination for mutual aid
10. Add integration with public alerting systems
11. Create a mobile app for field responders to update status
12. Implement automatic dispatch prioritization during mass casualty incidents
13. Add weather data integration to account for road conditions
"""
