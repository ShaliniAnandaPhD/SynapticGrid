/*
 * Data Aggregator (aggregator.rs)
 * 
 * PURPOSE:
 * This WebAssembly module provides efficient batch processing and aggregation of
 * high-frequency simulation events. It maintains a consistent state model and
 * performs complex metric calculations with minimal overhead.
 *
 * KEY FUNCTIONS:
 * - Maintains the latest state of all simulation nodes in memory
 * - Processes event batches to update the overall system state
 * - Performs efficient metric calculations through incremental updates
 * - Provides complete, consistent state snapshots for visualization
 * - Uses Rust's performance optimizations to handle thousands of nodes efficiently
 */

// aggregator.rs
use wasm_bindgen::prelude::*;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

#[wasm_bindgen]
pub struct DataAggregator {
    traffic_nodes: HashMap<String, TrafficNode>,
    energy_nodes: HashMap<String, EnergyNode>,
    waste_nodes: HashMap<String, WasteNode>,
    last_traffic_metrics: Option<TrafficMetrics>,
    last_energy_metrics: Option<EnergyMetrics>,
    last_waste_metrics: Option<WasteMetrics>,
}

#[derive(Serialize, Deserialize, Clone)]
struct TrafficNode {
    id: String,
    congestion: f32,
    flow: f32,
    position: Position,
    updated_at: u64,
}

#[derive(Serialize, Deserialize, Clone)]
struct EnergyNode {
    id: String,
    load: f32,
    capacity: f32,
    renewable: bool,
    position: Position,
    updated_at: u64,
}

#[derive(Serialize, Deserialize, Clone)]
struct WasteNode {
    id: String,
    fill_level: f32,
    capacity: f32,
    position: Position,
    updated_at: u64,
}

#[derive(Serialize, Deserialize, Clone, Copy)]
struct Position {
    x: f32,
    y: f32,
}

#[derive(Serialize, Deserialize, Clone)]
struct TrafficMetrics {
    average_congestion: f32,
    total_flow: f32,
    congestion_hotspots: u32,
}

#[derive(Serialize, Deserialize, Clone)]
struct EnergyMetrics {
    grid_load: f32,
    renewable_percentage: f32,
    stability: f32,
}

#[derive(Serialize, Deserialize, Clone)]
struct WasteMetrics {
    average_fill: f32,
    collection_efficiency: f32,
    overflow_count: u32,
}

#[derive(Serialize, Deserialize)]
struct AggregatedData {
    timestamp: u64,
    traffic: TrafficState,
    energy: EnergyState,
    waste: WasteState,
}

#[derive(Serialize, Deserialize)]
struct TrafficState {
    nodes: Vec<TrafficNode>,
    metrics: TrafficMetrics,
}

#[derive(Serialize, Deserialize)]
struct EnergyState {
    nodes: Vec<EnergyNode>,
    metrics: EnergyMetrics,
}

#[derive(Serialize, Deserialize)]
struct WasteState {
    nodes: Vec<WasteNode>,
    metrics: WasteMetrics,
}

#[wasm_bindgen]
impl DataAggregator {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        DataAggregator {
            traffic_nodes: HashMap::new(),
            energy_nodes: HashMap::new(),
            waste_nodes: HashMap::new(),
            last_traffic_metrics: None,
            last_energy_metrics: None,
            last_waste_metrics: None,
        }
    }
    
    #[wasm_bindgen]
    pub fn process_batch(&mut self, batch_json: &str) -> String {
        // Parse the incoming batch data
        let events: Vec<SimulationEvent> = match serde_json::from_str(batch_json) {
            Ok(data) => data,
            Err(_) => return "{}".to_string(),
        };
        
        // Process events by type
        for event in events {
            match event.event_type.as_str() {
                "traffic" => self.process_traffic_event(event),
                "energy" => self.process_energy_event(event),
                "waste" => self.process_waste_event(event),
                _ => {}
            }
        }
        
        // Generate aggregated data
        let aggregated = self.aggregate_data();
        
        // Serialize and return
        match serde_json::to_string(&aggregated) {
            Ok(json) => json,
            Err(_) => "{}".to_string(),
        }
    }
    
    fn process_traffic_event(&mut self, event: SimulationEvent) {
        if let Some(traffic_data) = event.traffic_data {
            // Update node states
            for node in traffic_data.nodes {
                self.traffic_nodes.insert(node.id.clone(), node);
            }
            
            // Update metrics if provided
            if let Some(metrics) = traffic_data.metrics {
                self.last_traffic_metrics = Some(metrics);
            }
        }
    }
    
    fn process_energy_event(&mut self, event: SimulationEvent) {
        if let Some(energy_data) = event.energy_data {
            // Update node states
            for node in energy_data.nodes {
                self.energy_nodes.insert(node.id.clone(), node);
            }
            
            // Update metrics if provided
            if let Some(metrics) = energy_data.metrics {
                self.last_energy_metrics = Some(metrics);
            }
        }
    }
    
    fn process_waste_event(&mut self, event: SimulationEvent) {
        if let Some(waste_data) = event.waste_data {
            // Update node states
            for node in waste_data.nodes {
                self.waste_nodes.insert(node.id.clone(), node);
            }
            
            // Update metrics if provided
            if let Some(metrics) = waste_data.metrics {
                self.last_waste_metrics = Some(metrics);
            }
        }
    }
    
    fn aggregate_data(&self) -> AggregatedData {
        // Calculate traffic metrics if not provided
        let traffic_metrics = self.last_traffic_metrics.clone().unwrap_or_else(|| {
            self.calculate_traffic_metrics()
        });
        
        // Calculate energy metrics if not provided
        let energy_metrics = self.last_energy_metrics.clone().unwrap_or_else(|| {
            self.calculate_energy_metrics()
        });
        
        // Calculate waste metrics if not provided
        let waste_metrics = self.last_waste_metrics.clone().unwrap_or_else(|| {
            self.calculate_waste_metrics()
        });
        
        AggregatedData {
            timestamp: current_time_ms(),
            traffic: TrafficState {
                nodes: self.traffic_nodes.values().cloned().collect(),
                metrics: traffic_metrics,
            },
            energy: EnergyState {
                nodes: self.energy_nodes.values().cloned().collect(),
                metrics: energy_metrics,
            },
            waste: WasteState {
                nodes: self.waste_nodes.values().cloned().collect(),
                metrics: waste_metrics,
            },
        }
    }
    
    fn calculate_traffic_metrics(&self) -> TrafficMetrics {
        let mut total_congestion = 0.0;
        let mut total_flow = 0.0;
        let mut hotspots = 0;
        
        for node in self.traffic_nodes.values() {
            total_congestion += node.congestion;
            total_flow += node.flow;
            
            if node.congestion > 0.8 {
                hotspots += 1;
            }
        }
        
        let count = self.traffic_nodes.len() as f32;
        let avg_congestion = if count > 0.0 { total_congestion / count } else { 0.0 };
        
        TrafficMetrics {
            average_congestion: avg_congestion,
            total_flow,
            congestion_hotspots: hotspots,
        }
    }
    
    fn calculate_energy_metrics(&self) -> EnergyMetrics {
        let mut total_load = 0.0;
        let mut total_capacity = 0.0;
        let mut renewable_capacity = 0.0;
        
        for node in self.energy_nodes.values() {
            total_load += node.load;
            total_capacity += node.capacity;
            
            if node.renewable {
                renewable_capacity += node.capacity;
            }
        }
        
        let renewable_percentage = if total_capacity > 0.0 {
            renewable_capacity / total_capacity
        } else {
            0.0
        };
        
        let grid_load = if total_capacity > 0.0 {
            total_load / total_capacity
        } else {
            0.0
        };
        
        // Stability is inversely related to how close we are to capacity
        let stability = 1.0 - (grid_load.min(1.0) * 0.8);
        
        EnergyMetrics {
            grid_load,
            renewable_percentage,
            stability,
        }
    }
    
    fn calculate_waste_metrics(&self) -> WasteMetrics {
        let mut total_fill = 0.0;
        let mut overflow_count = 0;
        
        for node in self.waste_nodes.values() {
            total_fill += node.fill_level;
            
            if node.fill_level > 0.9 {
                overflow_count += 1;
            }
        }
        
        let count = self.waste_nodes.len() as f32;
        let average_fill = if count > 0.0 { total_fill / count } else { 0.0 };
        
        // Collection efficiency is inversely related to average fill level
        let collection_efficiency = 1.0 - (average_fill * 0.8);
        
        WasteMetrics {
            average_fill,
            collection_efficiency,
            overflow_count,
        }
    }
}

// Helper function to get current time in milliseconds
fn current_time_ms() -> u64 {
    #[cfg(target_arch = "wasm32")]
    {
        js_sys::Date::now() as u64
    }
    
    #[cfg(not(target_arch = "wasm32"))]
    {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64
    }
}
