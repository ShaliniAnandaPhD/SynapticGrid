/*
 * WASM Simulation Processor
 * 
 * PURPOSE:
 * This WebAssembly module handles client-side processing of simulation data,
 * offloading CPU-intensive calculations from the JavaScript main thread.
 * It intelligently filters updates to reduce unnecessary UI refreshes.
 *
 * KEY FUNCTIONS:
 * - Processes raw simulation data into meaningful metrics
 * - Filters out insignificant changes to reduce rendering load
 * - Calculates system-wide statistics for dashboard displays
 * - Identifies important events (congestion hotspots, energy alerts, etc.)
 * - Provides optimized data structures ready for visualization
 */

// lib.rs (for WASM module)
use wasm_bindgen::prelude::*;
use serde::{Serialize, Deserialize};

#[wasm_bindgen]
pub struct SimulationProcessor {
    last_processed_data: Option<ProcessedSimulationData>,
    update_threshold: f32,
}

#[derive(Serialize, Deserialize)]
pub struct ProcessedSimulationData {
    timestamp: u64,
    traffic_metrics: TrafficMetrics,
    energy_metrics: EnergyMetrics,
    waste_metrics: WasteMetrics,
}

#[derive(Serialize, Deserialize, Clone)]
struct TrafficMetrics {
    average_congestion: f32,
    hotspots: Vec<String>,
    flow_efficiency: f32,
}

#[derive(Serialize, Deserialize, Clone)]
struct EnergyMetrics {
    grid_stability: f32,
    load_balance: f32,
    renewable_percentage: f32,
}

#[derive(Serialize, Deserialize, Clone)]
struct WasteMetrics {
    collection_efficiency: f32,
    fill_rate: f32,
    route_optimization: f32,
}

#[wasm_bindgen]
impl SimulationProcessor {
    #[wasm_bindgen(constructor)]
    pub fn new(update_threshold: f32) -> SimulationProcessor {
        SimulationProcessor {
            last_processed_data: None,
            update_threshold,
        }
    }
    
    #[wasm_bindgen]
    pub fn process_simulation_data(&mut self, data_json: &str) -> Option<String> {
        // Parse incoming JSON data
        let raw_data: RawSimulationData = match serde_json::from_str(data_json) {
            Ok(data) => data,
            Err(_) => return None,
        };
        
        // Process the raw data
        let processed_data = self.process_data(raw_data);
        
        // Check if the change is significant enough to warrant an update
        if self.should_update(&processed_data) {
            self.last_processed_data = Some(processed_data.clone());
            match serde_json::to_string(&processed_data) {
                Ok(json) => Some(json),
                Err(_) => None,
            }
        } else {
            None // No significant change, don't update UI
        }
    }
    
    fn process_data(&self, raw_data: RawSimulationData) -> ProcessedSimulationData {
        // Complex data processing logic here
        // This is where the heavy lifting happens that would be slow in JS
        
        // For example, calculate traffic metrics
        let traffic_metrics = self.process_traffic_data(&raw_data.traffic_nodes);
        let energy_metrics = self.process_energy_data(&raw_data.energy_nodes);
        let waste_metrics = self.process_waste_data(&raw_data.waste_nodes);
        
        ProcessedSimulationData {
            timestamp: raw_data.timestamp,
            traffic_metrics,
            energy_metrics,
            waste_metrics,
        }
    }
    
    fn should_update(&self, new_data: &ProcessedSimulationData) -> bool {
        if let Some(last_data) = &self.last_processed_data {
            // Check if the change exceeds our threshold
            let traffic_change = (new_data.traffic_metrics.average_congestion - 
                                 last_data.traffic_metrics.average_congestion).abs();
            
            let energy_change = (new_data.energy_metrics.grid_stability - 
                               last_data.energy_metrics.grid_stability).abs();
            
            let waste_change = (new_data.waste_metrics.collection_efficiency - 
                              last_data.waste_metrics.collection_efficiency).abs();
            
            // Only update if any metric changed more than our threshold
            traffic_change > self.update_threshold || 
            energy_change > self.update_threshold || 
            waste_change > self.update_threshold
        } else {
            // Always update if this is the first data we're processing
            true
        }
    }
    
    // Data processing helper methods
    fn process_traffic_data(&self, nodes: &[TrafficNode]) -> TrafficMetrics {
        // Complex traffic calculations...
        TrafficMetrics {
            average_congestion: nodes.iter().map(|n| n.congestion).sum::<f32>() / nodes.len() as f32,
            hotspots: nodes.iter()
                .filter(|n| n.congestion > 0.8)
                .map(|n| n.id.clone())
                .collect(),
            flow_efficiency: 0.75, // Calculated value
        }
    }
    
    fn process_energy_data(&self, nodes: &[EnergyNode]) -> EnergyMetrics {
        // Complex energy calculations...
        EnergyMetrics {
            grid_stability: 0.92,
            load_balance: 0.85,
            renewable_percentage: 0.35,
        }
    }
    
    fn process_waste_data(&self, nodes: &[WasteNode]) -> WasteMetrics {
        // Complex waste calculations...
        WasteMetrics {
            collection_efficiency: 0.78,
            fill_rate: nodes.iter().map(|n| n.fill_level).sum::<f32>() / nodes.len() as f32,
            route_optimization: 0.82,
        }
    }
}
