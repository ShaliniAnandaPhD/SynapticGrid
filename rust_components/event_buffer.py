"""
Event Buffer (event_buffer.py)

PURPOSE:
This Python module implements a high-performance event buffering system that sits
between the RL simulation and the WebSocket server. It aggregates high-frequency
events to reduce network load and provides intelligent event processing.

KEY FUNCTIONS:
- Buffers events over configurable time intervals to reduce message frequency
- Intelligently aggregates events by type (traffic, energy, waste)
- Handles type-specific aggregation logic for each subsystem
- Manages WebSocket connections with clients
- Processes parameter updates from clients back to the simulation
"""

# event_buffer.py (Python side for simulation system)
import asyncio
import json
import time
import websockets
from collections import defaultdict

class SimulationEventBuffer:
    def __init__(self, buffer_interval=0.1, max_buffer_size=100):
        self.buffer_interval = buffer_interval  # seconds
        self.max_buffer_size = max_buffer_size
        self.buffer = defaultdict(list)
        self.clients = set()
        self.running = False
        self.lock = asyncio.Lock()
    
    async def start(self):
        """Start the buffer processing loop"""
        self.running = True
        asyncio.create_task(self._process_buffer_loop())
        await self._start_websocket_server()
    
    async def stop(self):
        """Stop the buffer processing"""
        self.running = False
    
    async def add_event(self, system_type, event_data):
        """Add a new event to the buffer"""
        async with self.lock:
            self.buffer[system_type].append(event_data)
            
            # If buffer gets too large, process immediately
            if len(self.buffer[system_type]) >= self.max_buffer_size:
                await self._process_buffer()
    
    async def _process_buffer_loop(self):
        """Background task to periodically process the buffer"""
        while self.running:
            await asyncio.sleep(self.buffer_interval)
            await self._process_buffer()
    
    async def _process_buffer(self):
        """Process all events in the buffer and send to clients"""
        if not self.clients:
            # No clients connected, just clear the buffer
            async with self.lock:
                self.buffer.clear()
            return
        
        async with self.lock:
            if not any(self.buffer.values()):
                return  # No events to process
            
            # Aggregate events by type
            aggregated_data = {}
            for system_type, events in self.buffer.items():
                if not events:
                    continue
                
                # For each system type, aggregate in a type-specific way
                if system_type == "traffic":
                    aggregated_data[system_type] = self._aggregate_traffic_events(events)
                elif system_type == "energy":
                    aggregated_data[system_type] = self._aggregate_energy_events(events)
                elif system_type == "waste":
                    aggregated_data[system_type] = self._aggregate_waste_events(events)
            
            # Clear buffer after processing
            self.buffer.clear()
            
            # Create a complete message with all aggregated data
            message = {
                "timestamp": time.time(),
                "data": aggregated_data
            }
            
            # Send to all connected clients
            if self.clients:
                message_str = json.dumps(message)
                await asyncio.gather(
                    *[client.send(message_str) for client in self.clients],
                    return_exceptions=True
                )
    
    def _aggregate_traffic_events(self, events):
        """Aggregate traffic events into a single state"""
        # For traffic, we usually want the most recent state of each node
        latest_states = {}
        for event in events:
            for node in event.get("nodes", []):
                node_id = node.get("id")
                if node_id:
                    latest_states[node_id] = node
        
        return {
            "nodes": list(latest_states.values()),
            "global_metrics": events[-1].get("global_metrics", {})
        }
    
    def _aggregate_energy_events(self, events):
        """Aggregate energy grid events"""
        # Similar pattern to traffic aggregation
        latest_grid_states = {}
        for event in events:
            for node in event.get("grid_nodes", []):
                node_id = node.get("id")
                if node_id:
                    latest_grid_states[node_id] = node
        
        return {
            "grid_nodes": list(latest_grid_states.values()),
            "grid_metrics": events[-1].get("grid_metrics", {})
        }
    
    def _aggregate_waste_events(self, events):
        """Aggregate waste management events"""
        # For waste, we might care about cumulative metrics
        latest_bin_states = {}
        collection_metrics = {"bins_collected": 0, "distance_traveled": 0}
        
        for event in events:
            for bin_data in event.get("bins", []):
                bin_id = bin_data.get("id")
                if bin_id:
                    latest_bin_states[bin_id] = bin_data
            
            # Accumulate metrics
            metrics = event.get("collection_metrics", {})
            collection_metrics["bins_collected"] += metrics.get("bins_collected", 0)
            collection_metrics["distance_traveled"] += metrics.get("distance_traveled", 0)
        
        return {
            "bins": list(latest_bin_states.values()),
            "collection_metrics": collection_metrics
        }
    
    async def _start_websocket_server(self):
        """Start the WebSocket server for client connections"""
        async def handler(websocket, path):
            # Register new client
            self.clients.add(websocket)
            try:
                async for message in websocket:
                    # Handle incoming messages (parameter updates, etc.)
                    try:
                        data = json.loads(message)
                        if data.get("type") == "parameter_update":
                            # Forward to simulation controller
                            asyncio.create_task(self._handle_parameter_update(data))
                    except json.JSONDecodeError:
                        pass
            finally:
                #
