A Python-React application that simulates a decentralized smart grid with AI agents using reinforcement learning to optimize energy trading and distribution.

## Overview

This project provides a simulation of a smart energy grid where different nodes (solar homes, wind farms, batteries, and consumers) interact with each other through energy trading. Each node is controlled by an AI agent that makes decisions based on current grid conditions, weather, and learned patterns to optimize various objectives like profit, efficiency, or grid stability.

The application consists of:
- **Python Backend**: Simulates the smart grid network and AI agent behaviors
- **React Frontend**: Visualizes the grid, trades, and agent performance in real-time

## Features

- Multi-agent reinforcement learning system
- Real-time energy trading simulation 
- Visualization of node states, energy flows, and trades
- Dynamic weather conditions affecting renewable energy production
- Multiple agent types with different strategies:
  - **Optimizer Agent**: Maximizes energy efficiency and profit
  - **Forecaster Agent**: Predicts future energy needs and prices
  - **Trader Agent**: Negotiates and executes energy exchanges
  - **Balancer Agent**: Ensures grid stability and reliability
- Reinforcement learning metrics visualization
- Different policy types: Greedy, Exploratory, and Optimized

## Project Structure

```
smart_grid/
├── models/
│   ├── __init__.py
│   ├── node.py          # Node classes (SolarHome, WindFarm, etc.)
│   └── agent.py         # Agent classes for AI behavior
├── __init__.py
└── simulation.py        # Main simulation logic
app.py                   # Flask API server
frontend/                # React frontend application
└── src/
    ├── components/      # React components for dashboard
    ├── services/        # API service for backend communication
    └── App.js           # Main application component
requirements.txt         # Python dependencies
package.json             # Node.js dependencies
```

## Installation

### Backend Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/smart-grid-simulation.git
cd smart-grid-simulation
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the Flask server:
```bash
python app.py
```

The server will start at http://localhost:5000

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm start
```

The application will open in your default browser at http://localhost:3000

## Usage

- **Dashboard**: The main view shows the current state of the grid, energy statistics, and agent performance
- **Network View**: Visualizes nodes and energy trades between them
- **Node Details**: Click on any node to see detailed information and agent metrics
- **Training Mode**: Enable training mode to accelerate agent learning
- **Policy Selection**: Switch between different agent policy types

## Technical Details

### Node Types

- **Solar Home**: Produces energy based on time of day and weather conditions
- **Wind Farm**: Produces energy based on wind speed
- **Battery**: Stores or releases energy based on grid conditions
- **Consumer**: Consumes energy with patterns based on time of day

### Agent Behavior

Each agent uses a simplified reinforcement learning approach:
- Observes grid state and node conditions
- Makes decisions to maximize rewards
- Learns from outcomes to improve future decisions
- Updates Q-values for different actions
- Balances exploration and exploitation

### Simulation Parameters

- **Time Step**: 5 minutes per simulation step
- **Weather Model**: Dynamic temperature, cloud cover, and wind speed affecting production
- **Price Model**: Dynamic pricing based on supply/demand balance
- **Learning Rate**: Configurable learning rate for agents (default: 0.01-0.06)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with React and Flask
- Uses Recharts for data visualization
- Inspired by research in multi-agent reinforcement learning and energy microgrids
