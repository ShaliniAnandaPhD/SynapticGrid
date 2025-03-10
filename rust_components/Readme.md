# Real-Time RL Visualization Optimization System

This system provides a high-performance solution for visualizing real-time data from reinforcement learning simulations in a smart city context. It's specifically designed to handle high-frequency updates (100+ updates per second) while maintaining responsive UI performance.

## System Architecture

The solution consists of several coordinated components:

1. **Rust WebSocket Server** - Efficiently handles client connections and data streaming
2. **WebAssembly Data Processor** - Offloads CPU-intensive calculations from the main thread
3. **WebAssembly Data Aggregator** - Manages state and provides optimized data for visualization
4. **Python Event Buffer** - Intelligently batches and aggregates high-frequency events
5. **React Dashboard** - Provides the UI with optimized rendering strategies
6. **Visualization Hook** - Custom React hook for efficient canvas/SVG rendering

## Component Overview

### Rust WebSocket Server (`ws_server.rs`)

A high-performance WebSocket server built with Axum that efficiently handles thousands of concurrent connections:

- Maintains connection state for all connected clients
- Broadcasts simulation updates to all clients efficiently 
- Provides immediate state synchronization for new connections
- Handles parameter updates from clients back to the simulation

### WebAssembly Data Processor (`lib.rs`)

A WASM module that offloads data processing to a separate thread:

- Processes raw simulation data into meaningful metrics
- Filters out insignificant changes to reduce rendering load
- Calculates system-wide statistics for dashboard displays
- Identifies important events (congestion hotspots, energy alerts, etc.)

### WebAssembly Data Aggregator (`aggregator.rs`)

A WASM module that provides efficient batch processing of events:

- Maintains the latest state of all simulation nodes in memory
- Processes event batches to update the overall system state
- Performs efficient metric calculations through incremental updates
- Provides complete, consistent state snapshots for visualization

### Python Event Buffer (`event_buffer.py`)

A Python module that sits between the RL simulation and the WebSocket server:

- Buffers events over configurable time intervals to reduce message frequency
- Intelligently aggregates events by type (traffic, energy, waste)
- Handles type-specific aggregation logic for each subsystem
- Manages WebSocket connections with clients
- Processes parameter updates from clients back to the simulation

### React Dashboard (`SimulationDashboard.jsx`)

The main frontend visualization interface:

- Establishes and maintains WebSocket connections to the backend
- Processes incoming data through WebAssembly modules
- Throttles UI updates for consistent performance
- Implements memoization to prevent unnecessary re-renders
- Manages user parameter inputs and sends them to the simulation

### Visualization Hook (`useOptimizedVisualization.js`)

A custom React hook for efficient visualization rendering:

- Provides a unified API for both SVG and Canvas-based visualizations
- Implements performance optimizations like throttling and update thresholds
- Manages visualization lifecycle (initialization, updates, cleanup)
- Handles rendering and scaling for both SVG and Canvas contexts
- Efficiently updates only what's changed instead of re-rendering everything

## Performance Optimizations

This system implements several key optimizations:

1. **Data Processing**
   - Offloads heavy processing to WebAssembly
   - Implements efficient state management with HashMaps
   - Uses incremental updates to avoid reprocessing unchanged data

2. **Network Optimization**
   - Batches high-frequency events to reduce network traffic
   - Implements intelligent data aggregation to minimize payload size
   - Uses non-blocking I/O for all network operations

3. **Rendering Optimization**
   - Implements intelligent update throttling
   - Uses change threshold detection to skip insignificant updates
   - Chooses optimal rendering strategy (Canvas vs. SVG) based on data size
   - Memoizes React components to prevent unnecessary re-renders

4. **Memory Efficiency**
   - Maintains only the latest state in memory
   - Implements efficient data structures
   - Uses Rust's zero-cost abstractions for optimal performance

## Getting Started

### Prerequisites

- Rust (latest stable version)
- Node.js (v14+)
- Python 3.7+
- wasm-pack

### Building the Server

```bash
# Build the WebSocket server
cd server
cargo build --release

# Run the server
./target/release/ws_server
```

### Building the WebAssembly Modules

```bash
# Build the simulation processor WASM module
cd wasm-processor
wasm-pack build --target web

# Build the data aggregator WASM module
cd wasm-aggregator
wasm-pack build --target web
```

### Running the Frontend

```bash
# Install dependencies
cd frontend
npm install

# Start the development server
npm run dev
```

### Starting the Event Buffer

```bash
# Install dependencies
cd event-buffer
pip install -r requirements.txt

# Run the event buffer
python event_buffer.py
```

## Performance Recommendations

For optimal performance:

1. **GPU Acceleration**: Enable GPU acceleration for visualizations by using WebGL
2. **Web Workers**: Offload additional processing to Web Workers when possible
3. **Adaptive Detail Level**: Implement level-of-detail rendering that adjusts visualization complexity based on performance metrics
4. **React Optimization**: Use `React.memo` for all visualization components and carefully implement `shouldComponentUpdate`
5. **Monitoring**: Add performance monitoring to track render times, network latency, and memory usage

## License

[MIT License](LICENSE)
