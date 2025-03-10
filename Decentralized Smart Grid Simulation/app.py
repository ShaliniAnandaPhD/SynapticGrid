from flask import Flask, jsonify, request
from flask_cors import CORS
import threading
import time
from smart_grid.simulation import SmartGridSimulation

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Create simulation instance
simulation = SmartGridSimulation()

# Flag to control simulation thread
simulation_running = True

# Simulation update interval in seconds
UPDATE_INTERVAL = 2.0

def simulation_thread():
    """Background thread that updates the simulation."""
    global simulation_running
    
    while simulation_running:
        simulation.update()
        time.sleep(UPDATE_INTERVAL)

@app.route('/api/grid/state', methods=['GET'])
def get_grid_state():
    """Return the current state of the grid."""
    return jsonify(simulation.get_full_state())

@app.route('/api/grid/node/<int:node_id>/metrics', methods=['GET'])
def get_node_metrics(node_id):
    """Return RL metrics for a specific node."""
    metrics = simulation.get_node_rl_metrics(node_id)
    if metrics:
        return jsonify(metrics)
    return jsonify({'error': 'Node not found'}), 404

@app.route('/api/grid/control', methods=['POST'])
def control_simulation():
    """Control the simulation settings."""
    data = request.json
    
    if 'running' in data:
        global simulation_running
        simulation_running = data['running']
        return jsonify({'status': 'success', 'running': simulation_running})
    
    return jsonify({'error': 'Invalid control command'}), 400

if __name__ == '__main__':
    # Start simulation in a background thread
    sim_thread = threading.Thread(target=simulation_thread, daemon=True)
    sim_thread.start()
    
    # Start Flask server
    app.run(debug=True, host='0.0.0.0', port=5000)
