/*
 * React Dashboard Component (SimulationDashboard.jsx)
 * 
 * PURPOSE:
 * This React component serves as the main frontend visualization interface for
 * the RL simulation data. It handles WebSocket communication, processes data
 * through WebAssembly, and efficiently renders real-time updates.
 *
 * KEY FUNCTIONS:
 * - Establishes and maintains WebSocket connections to the backend
 * - Processes incoming data through WebAssembly modules
 * - Throttles UI updates for consistent performance
 * - Implements memoization to prevent unnecessary re-renders
 * - Manages user parameter inputs and sends them to the simulation
 * - Provides responsive, component-based visualizations for each subsystem
 */

// SimulationDashboard.jsx
import React, { useState, useEffect, useRef, useMemo } from 'react';
import { useWebSocket } from 'react-use-websocket';
import * as d3 from 'd3';
import { init, getInstance } from './wasmInit'; // Helper to initialize WASM

const SimulationDashboard = () => {
  const [simulationData, setSimulationData] = useState(null);
  const [parameters, setParameters] = useState({
    learningRate: 0.001,
    discountFactor: 0.99,
    explorationRate: 0.1,
  });
  const [updateThreshold, setUpdateThreshold] = useState(0.05);
  const processorRef = useRef(null);
  
  // Keep track of last UI update time for throttling
  const lastUpdateRef = useRef(0);
  
  // Setup WebSocket connection
  const { sendMessage, lastMessage } = useWebSocket('ws://localhost:3001/ws', {
    shouldReconnect: () => true,
  });
  
  // Initialize WASM module
  useEffect(() => {
    const initWasm = async () => {
      await init();
      const wasmModule = getInstance();
      processorRef.current = new wasmModule.SimulationProcessor(updateThreshold);
    };
    
    initWasm();
    
    return () => {
      // Clean up WASM resources if needed
      if (processorRef.current) {
        processorRef.current.free();
      }
    };
  }, [updateThreshold]);
  
  // Process incoming WebSocket messages through WASM
  useEffect(() => {
    if (!lastMessage || !processorRef.current) return;
    
    try {
      // Process data through WASM
      const processedData = processorRef.current.process_simulation_data(lastMessage.data);
      
      // Only update if there's significant change
      if (processedData) {
        const now = performance.now();
        
        // Limit updates to max 30 per second (33ms)
        if (now - lastUpdateRef.current > 33) {
          setSimulationData(JSON.parse(processedData));
          lastUpdateRef.current = now;
        }
      }
    } catch (error) {
      console.error('Error processing simulation data:', error);
    }
  }, [lastMessage]);
  
  // Send parameter updates to the server
  const updateParameters = (newParams) => {
    setParameters(newParams);
    sendMessage(JSON.stringify({ 
      type: 'parameter_update',
      parameters: newParams
    }));
  };
  
  // Use React.memo for child components to prevent unnecessary re-renders
  const TrafficView = useMemo(() => {
    if (!simulationData) return null;
    
    return (
      <div className="panel">
        <h2>Traffic System</h2>
        <div className="metrics">
          <div className="metric">
            <span>Avg. Congestion:</span>
            <span>{(simulationData.traffic_metrics.average_congestion * 100).toFixed(1)}%</span>
          </div>
          <div className="metric">
            <span>Flow Efficiency:</span>
            <span>{(simulationData.traffic_metrics.flow_efficiency * 100).toFixed(1)}%</span>
          </div>
        </div>
        {/* Traffic visualization would go here */}
      </div>
    );
  }, [simulationData?.traffic_metrics]);
  
  const EnergyView = useMemo(() => {
    if (!simulationData) return null;
    
    return (
      <div className="panel">
        <h2>Energy Grid</h2>
        <div className="metrics">
          <div className="metric">
            <span>Grid Stability:</span>
            <span>{(simulationData.energy_metrics.grid_stability * 100).toFixed(1)}%</span>
          </div>
          <div className="metric">
            <span>Renewable Usage:</span>
            <span>{(simulationData.energy_metrics.renewable_percentage * 100).toFixed(1)}%</span>
          </div>
        </div>
        {/* Energy visualization would go here */}
      </div>
    );
  }, [simulationData?.energy_metrics]);
  
  const WasteView = useMemo(() => {
    if (!simulationData) return null;
    
    return (
      <div className="panel">
        <h2>Waste Management</h2>
        <div className="metrics">
          <div className="metric">
            <span>Collection Efficiency:</span>
            <span>{(simulationData.waste_metrics.collection_efficiency * 100).toFixed(1)}%</span>
          </div>
          <div className="metric">
            <span>Average Fill Rate:</span>
            <span>{(simulationData.waste_metrics.fill_rate * 100).toFixed(1)}%</span>
          </div>
        </div>
        {/* Waste visualization would go here */}
      </div>
    );
  }, [simulationData?.waste_metrics]);
  
  return (
    <div className="simulation-dashboard">
      <div className="controls">
        <h2>RL Parameters</h2>
        <div className="parameter-sliders">
          <label>
            Learning Rate:
            <input
              type="range"
              min="0.0001"
              max="0.01"
              step="0.0001"
              value={parameters.learningRate}
              onChange={(e) => updateParameters({
                ...parameters,
                learningRate: parseFloat(e.target.value)
              })}
            />
            <span>{parameters.learningRate}</span>
          </label>
          
          <label>
            Discount Factor:
            <input
              type="range"
              min="0.8"
              max="0.99"
              step="0.01"
              value={parameters.discountFactor}
              onChange={(e) => updateParameters({
                ...parameters,
                discountFactor: parseFloat(e.target.value)
              })}
            />
            <span>{parameters.discountFactor}</span>
          </label>
          
          <label>
            Exploration Rate:
            <input
              type="range"
              min="0.01"
              max="0.3"
              step="0.01"
              value={parameters.explorationRate}
              onChange={(e) => updateParameters({
                ...parameters,
                explorationRate: parseFloat(e.target.value)
              })}
            />
            <span>{parameters.explorationRate}</span>
          </label>
        </div>
        
        <h3>Display Settings</h3>
        <label>
          Update Threshold:
          <input
            type="range"
            min="0.01"
            max="0.2"
            step="0.01"
            value={updateThreshold}
            onChange={(e) => setUpdateThreshold(parseFloat(e.target.value))}
          />
          <span>{updateThreshold}</span>
        </label>
      </div>
      
      <div className="visualization-panels">
        {TrafficView}
        {EnergyView}
        {WasteView}
      </div>
    </div>
  );
};

export default React.memo(SimulationDashboard);
