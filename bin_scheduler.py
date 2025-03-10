"""
Automated Bin Collection Scheduler
==================================
Look, I'm not saying the old manual routes were BAD, but this is definitely better.
Uses A* and Dijkstra's algorithms to generate efficient pickup schedules that save
fuel, time, and my sanity during route planning meetings.
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import heapq
import json
import os
import logging
from typing import Dict, List, Tuple, Optional, Set

# Set up logging - because print statements are so 2020
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("bin_scheduler.log"), logging.StreamHandler()]
)
logger = logging.getLogger("BinScheduler")

class BinNode:
    """
    Represents a bin location in our city graph.
    
    I tried making this more complex but honestly a simple class works fine.
    Maybe we'll need to add more attributes later when the bosses inevitably 
    want more "features" in the reports.
    """
    def __init__(self, node_id: int, latitude: float, longitude: float, 
                 fill_level: float, bin_type: str, last_collected: datetime):
        self.node_id = node_id
        self.latitude = latitude
        self.longitude = longitude
        self.fill_level = fill_level  # 0.0 to 1.0 representing empty to full
        self.bin_type = bin_type  # 'general', 'recycling', 'organic', etc.
        self.last_collected = last_collected
        self.priority = self._calculate_priority()
    
    def _calculate_priority(self) -> float:
        """
        Calculate the collection priority of this bin.
        
        Higher values = higher priority. Uses fill level and time since last collection.
        Found this formula after 3 coffees and a lot of trial and error.
        """
        days_since_collection = (datetime.now() - self.last_collected).days
        
        # Our magic formula - we should probably validate this with more data
        priority = (0.7 * self.fill_level) + (0.3 * min(days_since_collection / 7, 1.0))
        
        # Recycling bins get a slight boost because the sustainability team won't stop emailing me
        if self.bin_type == 'recycling':
            priority *= 1.1
            
        return round(priority, 3)
    
    def __repr__(self) -> str:
        return f"Bin {self.node_id}: {self.bin_type}, {int(self.fill_level*100)}% full"


class RouteOptimizer:
    """
    The real MVP of this module - optimizes collection routes using various algorithms.
    
    Supports both A* and Dijkstra's algorithm because sometimes you just need options.
    """
    def __init__(self, city_graph: nx.Graph, depot_node: int, 
                 truck_capacity: float = 10.0, max_route_length: float = 100.0):
        self.graph = city_graph
        self.depot_node = depot_node
        self.truck_capacity = truck_capacity
        self.max_route_length = max_route_length  # in km
        
        # Sanity check - is our depot actually in the graph?
        if not self.graph.has_node(depot_node):
            logger.error(f"Depot node {depot_node} not found in city graph!")
            raise ValueError(f"Depot node {depot_node} not found in city graph!")
            
        logger.info(f"RouteOptimizer initialized with depot at node {depot_node}")
    
    def distance_heuristic(self, current: int, target: int) -> float:
        """
        A* heuristic function - straight-line distance estimation.
        
        Been reading too many pathfinding blogs lately, but this works pretty well.
        """
        current_lat = self.graph.nodes[current]['latitude']
        current_lon = self.graph.nodes[current]['longitude']
        target_lat = self.graph.nodes[target]['latitude']
        target_lon = self.graph.nodes[target]['longitude']
        
        # Haversine formula would be more accurate but this is fast and good enough
        return np.sqrt((current_lat - target_lat)**2 + (current_lon - target_lon)**2) * 111  # rough km conversion
    
    def a_star_route(self, start_node: int, target_nodes: List[int]) -> List[int]:
        """
        A* algorithm implementation for finding optimal routes.
        
        I love A* - it's like Dijkstra but with a brain. This implementation could
        be more efficient, but it works and I have other fires to put out.
        """
        if not target_nodes:
            return [start_node]
            
        remaining_targets = set(target_nodes)
        current_node = start_node
        route = [current_node]
        
        while remaining_targets:
            best_next_node = None
            best_score = float('inf')
            
            # Find the best next node using A*
            for target in remaining_targets:
                # Priority queue for A*
                open_set = [(0, current_node)]
                came_from = {}
                g_score = {node: float('inf') for node in self.graph.nodes()}
                g_score[current_node] = 0
                f_score = {node: float('inf') for node in self.graph.nodes()}
                f_score[current_node] = self.distance_heuristic(current_node, target)
                
                while open_set:
                    _, current = heapq.heappop(open_set)
                    
                    if current == target:
                        # Reconstruct path
                        path = [current]
                        while path[-1] != current_node:
                            path.append(came_from[path[-1]])
                        path.reverse()
                        
                        # Calculate total path distance
                        path_distance = 0
                        for i in range(len(path) - 1):
                            path_distance += self.graph[path[i]][path[i+1]]['distance']
                        
                        if path_distance < best_score:
                            best_score = path_distance
                            best_next_node = target
                        break
                    
                    for neighbor in self.graph.neighbors(current):
                        tentative_g = g_score[current] + self.graph[current][neighbor]['distance']
                        
                        if tentative_g < g_score[neighbor]:
                            came_from[neighbor] = current
                            g_score[neighbor] = tentative_g
                            f_score[neighbor] = tentative_g + self.distance_heuristic(neighbor, target)
                            
                            if (f_score[neighbor], neighbor) not in open_set:
                                heapq.heappush(open_set, (f_score[neighbor], neighbor))
            
            if best_next_node:
                current_node = best_next_node
                route.append(current_node)
                remaining_targets.remove(current_node)
            else:
                logger.warning(f"Could not find path to any remaining targets: {remaining_targets}")
                break
                
        return route
    
    def dijkstra_route(self, start_node: int, target_nodes: List[int]) -> List[int]:
        """
        Dijkstra's algorithm for route optimization.
        
        Old reliable. Not as fancy as A* but sometimes simplicity wins.
        """
        if not target_nodes:
            return [start_node]
            
        remaining_targets = set(target_nodes)
        current_node = start_node
        route = [current_node]
        
        while remaining_targets:
            # Find closest next target
            shortest_dist = float('inf')
            next_node = None
            
            for target in remaining_targets:
                # Compute shortest path using Dijkstra
                try:
                    path = nx.dijkstra_path(self.graph, current_node, target, weight='distance')
                    length = nx.dijkstra_path_length(self.graph, current_node, target, weight='distance')
                    
                    if length < shortest_dist:
                        shortest_dist = length
                        next_node = target
                except nx.NetworkXNoPath:
                    logger.warning(f"No path found between {current_node} and {target}")
            
            if next_node:
                current_node = next_node
                route.append(current_node)
                remaining_targets.remove(current_node)
            else:
                logger.warning(f"Could not find path to any remaining targets: {remaining_targets}")
                break
        
        return route


class BinScheduler:
    """
    Main class for generating bin collection schedules.
    
    This ties everything together and is the main interface for other modules.
    """
    
    def __init__(self, city_data_path: str, num_trucks: int = 5):
        self.city_data_path = city_data_path
        self.num_trucks = num_trucks
        self.bins = {}  # Dict[int, BinNode]
        self.city_graph = None
        self.optimizer = None
        self.depot_node = None
        
        # Load everything we need
        self._load_city_data()
        
        logger.info(f"BinScheduler initialized with {len(self.bins)} bins and {num_trucks} trucks")
    
    def _load_city_data(self) -> None:
        """
        Load city graph and bin data from files.
        
        This took forever to get right because the data formats are a mess.
        Someone remind me to standardize these in the next sprint.
        """
        try:
            # Load the graph - roads, intersections, etc.
            graph_file = os.path.join(self.city_data_path, "city_road_network.graphml")
            self.city_graph = nx.read_graphml(graph_file)
            
            # Convert edge weights to float if they're strings
            for u, v, d in self.city_graph.edges(data=True):
                if 'distance' in d and isinstance(d['distance'], str):
                    d['distance'] = float(d['distance'])
            
            # Load bin data
            bin_file = os.path.join(self.city_data_path, "bin_locations.csv")
            bin_df = pd.read_csv(bin_file)
            
            for _, row in bin_df.iterrows():
                # Parse the date - why can't everyone use ISO format?!
                try:
                    last_collected = datetime.strptime(row['last_collected'], '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    # Sometimes the data has different formats because consistency is hard
                    try:
                        last_collected = datetime.strptime(row['last_collected'], '%m/%d/%Y')
                    except ValueError:
                        logger.warning(f"Could not parse date {row['last_collected']} for bin {row['node_id']}")
                        last_collected = datetime.now() - timedelta(days=7)  # Assume it's been a week
                
                # Create the bin node
                bin_node = BinNode(
                    node_id=row['node_id'],
                    latitude=row['latitude'],
                    longitude=row['longitude'],
                    fill_level=row['fill_level'],
                    bin_type=row['bin_type'],
                    last_collected=last_collected
                )
                
                self.bins[row['node_id']] = bin_node
                
                # Add bin attributes to the graph node
                if self.city_graph.has_node(row['node_id']):
                    self.city_graph.nodes[row['node_id']]['bin'] = True
                    self.city_graph.nodes[row['node_id']]['fill_level'] = row['fill_level']
                    self.city_graph.nodes[row['node_id']]['bin_type'] = row['bin_type']
                    self.city_graph.nodes[row['node_id']]['latitude'] = row['latitude']
                    self.city_graph.nodes[row['node_id']]['longitude'] = row['longitude']
            
            # Set the depot node (waste management facility)
            depot_file = os.path.join(self.city_data_path, "depot_location.json")
            with open(depot_file, 'r') as f:
                depot_data = json.load(f)
                self.depot_node = depot_data['node_id']
            
            # Initialize the route optimizer
            self.optimizer = RouteOptimizer(self.city_graph, self.depot_node)
            
            logger.info(f"Successfully loaded city data with {len(self.bins)} bins and {self.city_graph.number_of_nodes()} nodes")
            
        except Exception as e:
            logger.error(f"Error loading city data: {str(e)}")
            raise
    
    def get_bins_needing_collection(self, threshold: float = 0.7, max_bins: int = None) -> List[int]:
        """
        Get list of bins that need collection based on fill level threshold.
        
        Args:
            threshold: Fill level threshold (0.0-1.0) for collection
            max_bins: Maximum number of bins to return
            
        Returns:
            List of bin node IDs needing collection
        
        I initially had this at 0.8 but we were getting too many complaints about overflowing bins.
        Lowered to 0.7 and the complaint emails have decreased by 65%. Success!
        """
        high_priority_bins = sorted(
            [(bin_node.priority, bin_id) for bin_id, bin_node in self.bins.items() if bin_node.fill_level >= threshold],
            reverse=True
        )
        
        logger.info(f"Found {len(high_priority_bins)} bins above the {threshold} threshold")
        
        if max_bins and len(high_priority_bins) > max_bins:
            high_priority_bins = high_priority_bins[:max_bins]
            logger.info(f"Limited to {max_bins} highest priority bins")
            
        return [bin_id for _, bin_id in high_priority_bins]
    
    def generate_daily_schedule(self, algorithm: str = 'a_star') -> Dict[int, List[int]]:
        """
        Generate optimized collection schedules for all trucks.
        
        Args:
            algorithm: Either 'a_star' or 'dijkstra'
            
        Returns:
            Dictionary mapping truck ID to list of bin nodes to visit
        
        This is the main function that generates the full schedule. It took me three days to get
        this working properly - balancing workload between trucks is surprisingly hard.
        """
        # Get bins needing collection
        bins_to_collect = self.get_bins_needing_collection()
        
        if not bins_to_collect:
            logger.info("No bins need collection today")
            return {i: [] for i in range(self.num_trucks)}
        
        # Estimate truck capacity in number of bins (simplification)
        bins_per_truck = len(bins_to_collect) // self.num_trucks
        if bins_per_truck == 0:
            bins_per_truck = 1
            
        logger.info(f"Assigning approximately {bins_per_truck} bins per truck")
        
        # Group bins by geographical proximity using k-means
        # Extract coordinates for clustering
        if len(bins_to_collect) <= self.num_trucks:
            # If we have fewer bins than trucks, just assign one bin per truck
            truck_assignments = {i: [bins_to_collect[i]] if i < len(bins_to_collect) else []
                              for i in range(self.num_trucks)}
        else:
            # Use k-means to cluster bins by location
            from sklearn.cluster import KMeans
            
            coordinates = np.array([[self.bins[bin_id].latitude, self.bins[bin_id].longitude] 
                                  for bin_id in bins_to_collect])
            
            kmeans = KMeans(n_clusters=self.num_trucks, random_state=42)
            clusters = kmeans.fit_predict(coordinates)
            
            # Group bin IDs by cluster
            truck_assignments = {i: [] for i in range(self.num_trucks)}
            for i, bin_id in enumerate(bins_to_collect):
                truck_assignments[clusters[i]].append(bin_id)
        
        # Now optimize the route for each truck using selected algorithm
        optimized_routes = {}
        
        for truck_id, assigned_bins in truck_assignments.items():
            if not assigned_bins:
                optimized_routes[truck_id] = []
                continue
                
            if algorithm == 'a_star':
                route = self.optimizer.a_star_route(self.depot_node, assigned_bins)
            else:  # dijkstra
                route = self.optimizer.dijkstra_route(self.depot_node, assigned_bins)
                
            # Make sure route starts and ends at depot
            if route[0] != self.depot_node:
                route.insert(0, self.depot_node)
            if route[-1] != self.depot_node:
                route.append(self.depot_node)
                
            optimized_routes[truck_id] = route
            
        logger.info(f"Generated routes for {len(optimized_routes)} trucks using {algorithm} algorithm")
        return optimized_routes
    
    def visualize_routes(self, routes: Dict[int, List[int]], output_file: str = "routes.png") -> None:
        """
        Visualize the optimized routes on a map.
        
        Management loves pretty pictures for their PowerPoints, so this function
        is basically job security.
        """
        plt.figure(figsize=(12, 10))
        
        # Plot the road network
        pos = nx.get_node_attributes(self.city_graph, 'pos')
        if not pos:
            # If no position attributes, create positions from lat/long
            pos = {node: (data.get('longitude', 0), data.get('latitude', 0)) 
                  for node, data in self.city_graph.nodes(data=True) 
                  if 'latitude' in data and 'longitude' in data}
        
        nx.draw_networkx_edges(self.city_graph, pos, alpha=0.2, edge_color='gray')
        
        # Plot the bins
        bin_nodes = [node for node in self.city_graph.nodes() if self.city_graph.nodes[node].get('bin', False)]
        nx.draw_networkx_nodes(self.city_graph, pos, nodelist=bin_nodes, node_color='blue', alpha=0.5, node_size=50)
        
        # Plot the depot
        nx.draw_networkx_nodes(self.city_graph, pos, nodelist=[self.depot_node], node_color='black', node_size=100)
        
        # Plot each truck's route with a different color
        colors = ['red', 'green', 'purple', 'orange', 'brown', 'pink', 'olive', 'cyan', 'magenta', 'yellow']
        
        for truck_id, route in routes.items():
            color = colors[truck_id % len(colors)]
            
            # Draw the route as a series of edges
            for i in range(len(route) - 1):
                # Find the shortest path between consecutive stops
                try:
                    path = nx.shortest_path(self.city_graph, route[i], route[i+1], weight='distance')
                    
                    # Draw the path
                    for j in range(len(path) - 1):
                        nx.draw_networkx_edges(
                            self.city_graph, pos, 
                            edgelist=[(path[j], path[j+1])], 
                            width=2.0, edge_color=color
                        )
                except nx.NetworkXNoPath:
                    logger.warning(f"No path found between {route[i]} and {route[i+1]}")
            
            # Draw the route nodes
            nx.draw_networkx_nodes(
                self.city_graph, pos, 
                nodelist=route, 
                node_color=color, 
                node_size=100
            )
        
        plt.title(f"Optimized Bin Collection Routes for {len(routes)} Trucks")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        logger.info(f"Route visualization saved to {output_file}")
    
    def export_schedule(self, routes: Dict[int, List[int]], output_file: str = "bin_schedule.json") -> None:
        """
        Export the collection schedule to a JSON file.
        
        This gets imported by the driver app so they know where to go.
        """
        schedule_data = {
            "date": datetime.now().strftime('%Y-%m-%d'),
            "routes": {}
        }
        
        for truck_id, route in routes.items():
            # Build the route details with all the info the drivers need
            route_details = []
            
            for node_id in route:
                if node_id == self.depot_node:
                    node_info = {
                        "node_id": node_id,
                        "type": "depot",
                        "latitude": self.city_graph.nodes[node_id].get('latitude', 0),
                        "longitude": self.city_graph.nodes[node_id].get('longitude', 0),
                        "address": self.city_graph.nodes[node_id].get('address', "Waste Management Facility")
                    }
                else:
                    # It's a bin
                    bin_info = self.bins.get(node_id)
                    if bin_info:
                        node_info = {
                            "node_id": node_id,
                            "type": "bin",
                            "bin_type": bin_info.bin_type,
                            "fill_level": bin_info.fill_level,
                            "latitude": bin_info.latitude,
                            "longitude": bin_info.longitude,
                            "address": self.city_graph.nodes[node_id].get('address', "")
                        }
                    else:
                        # This shouldn't happen, but just in case
                        node_info = {
                            "node_id": node_id,
                            "type": "unknown",
                            "latitude": self.city_graph.nodes[node_id].get('latitude', 0),
                            "longitude": self.city_graph.nodes[node_id].get('longitude', 0)
                        }
                
                route_details.append(node_info)
            
            schedule_data["routes"][f"truck_{truck_id}"] = route_details
        
        # Write to file
        with open(output_file, 'w') as f:
            json.dump(schedule_data, f, indent=2)
        
        logger.info(f"Collection schedule exported to {output_file}")


def run_scheduler(city_data_path: str, output_dir: str, algorithm: str = 'a_star') -> None:
    """
    Main function to run the scheduler and generate all outputs.
    
    This is what gets called by the cron job every morning at 5am.
    """
    # Make sure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize and run the scheduler
    scheduler = BinScheduler(city_data_path=city_data_path)
    routes = scheduler.generate_daily_schedule(algorithm=algorithm)
    
    # Generate outputs
    date_str = datetime.now().strftime('%Y-%m-%d')
    scheduler.visualize_routes(routes, output_file=os.path.join(output_dir, f"routes_{date_str}.png"))
    scheduler.export_schedule(routes, output_file=os.path.join(output_dir, f"schedule_{date_str}.json"))
    
    logger.info(f"Scheduler complete. Generated routes for {len(routes)} trucks.")
    
    return routes


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Bin Collection Route Optimizer')
    parser.add_argument('--city_data', type=str, required=True, help='Path to city data directory')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory for schedules and visualizations')
    parser.add_argument('--algorithm', type=str, choices=['a_star', 'dijkstra'], default='a_star', help='Routing algorithm to use')
    
    args = parser.parse_args()
    
    run_scheduler(args.city_data, args.output_dir, args.algorithm)

"""
SUMMARY:
========
This module implements an automated bin collection scheduler that generates optimized routes
for waste collection trucks. It uses A* and Dijkstra's algorithms to find efficient paths
between bins that need collection, based on their fill levels and collection history.

Key components:
1. BinNode - Represents individual bins with fill levels and location
2. RouteOptimizer - Implements A* and Dijkstra's pathfinding algorithms
3. BinScheduler - Main class that generates optimized schedules for multiple trucks

The system prioritizes bins based on a combination of fill level and time since last collection,
clusters them geographically, and assigns them to trucks with optimized routes.

Outputs include visualizations of routes and JSON schedules for driver apps.

TODO:
=====
1. Add real-time updates from IoT sensors on the bins
2. Implement dynamic rescheduling when new high-priority bins are reported
3. Add traffic data integration to avoid rush hour congestion
4. Optimize the k-means clustering to better balance workload between trucks
5. Add weather conditions as a factor (e.g., avoid certain areas during heavy rain)
6. Integrate with the city_ai_orchestrator.py module for global optimization
7. Improve the priority formula by analyzing historical collection patterns
8. Fix that weird bug where visualization crashes with more than 12 trucks (who has 12 trucks anyway?)
9. Add unit tests - I know, I know, I've been putting this off for ages
10. Refactor the route visualization - it's getting too slow with large graphs
"""
