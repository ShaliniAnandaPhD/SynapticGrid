/*
 * WebSocket Server 
 * 
 * PURPOSE:
 * This high-performance WebSocket server acts as the communication bridge between
 * the RL simulation backend and the frontend visualization. It efficiently handles
 * real-time data streaming with support for thousands of concurrent connections.
 *
 * KEY FUNCTIONS:
 * - Maintains connection state for all connected clients
 * - Broadcasts simulation updates to all clients efficiently
 * - Provides immediate state synchronization for new connections
 * - Handles parameter updates from clients back to the simulation
 * - Uses Axum and Tokio for asynchronous, non-blocking I/O
 */

// ws_server.rs
use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::IntoResponse,
    routing::get,
    Router,
};
use futures::{SinkExt, StreamExt};
use std::{collections::HashMap, sync::Arc};
use tokio::sync::{mpsc, RwLock};
use tower_http::cors::CorsLayer;

// User connection state
struct AppState {
    // Map from user ID to sender channel
    users: RwLock<HashMap<String, mpsc::Sender<Message>>>,
    // Store latest simulation state to send to new connections
    latest_state: RwLock<SimulationState>,
}

// Simplified simulation state structure
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct SimulationState {
    timestamp: u64,
    traffic_data: Vec<TrafficNode>,
    energy_data: Vec<EnergyNode>,
    waste_data: Vec<WasteNode>,
    // Additional fields as needed
}

// Handlers for different node types
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct TrafficNode {
    id: String,
    congestion: f32,
    // Other traffic-specific fields
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct EnergyNode {
    id: String,
    load: f32,
    // Other energy-specific fields
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct WasteNode {
    id: String,
    fill_level: f32,
    // Other waste-specific fields
}

async fn websocket_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_socket(socket, state))
}

async fn handle_socket(socket: WebSocket, state: Arc<AppState>) {
    let (mut sender, mut receiver) = socket.split();
    
    // Generate a unique user ID
    let user_id = uuid::Uuid::new_v4().to_string();
    
    // Create a channel for sending messages to this user
    let (tx, mut rx) = mpsc::channel::<Message>(100);
    
    // Store the sender in our user map
    state.users.write().await.insert(user_id.clone(), tx);
    
    // Send the latest state immediately to new connections
    let latest_state = state.latest_state.read().await.clone();
    let state_json = serde_json::to_string(&latest_state).unwrap();
    let _ = sender.send(Message::Text(state_json)).await;
    
    // Handle incoming messages
    let mut recv_task = tokio::spawn(async move {
        while let Some(Ok(msg)) = receiver.next().await {
            match msg {
                Message::Text(text) => {
                    // Handle parameter updates from client
                    if let Ok(params) = serde_json::from_str::<SimulationParameters>(&text) {
                        // Forward parameters to simulation controller
                        // ...
                    }
                }
                Message::Close(_) => break,
                _ => {}
            }
        }
        
        // Remove user when connection closes
        state.users.write().await.remove(&user_id);
    });
    
    // Forward messages from the channel to the WebSocket
    let mut send_task = tokio::spawn(async move {
        while let Some(msg) = rx.recv().await {
            if sender.send(msg).await.is_err() {
                break;
            }
        }
    });
    
    // Wait for either task to finish
    tokio::select! {
        _ = (&mut recv_task) => send_task.abort(),
        _ = (&mut send_task) => recv_task.abort(),
    }
}

// Handler for simulation updates (called by your RL system)
async fn update_simulation_state(
    state: Arc<AppState>,
    new_state: SimulationState,
) {
    // Update the latest state
    *state.latest_state.write().await = new_state.clone();
    
    // Serialize the state once
    let state_json = serde_json::to_string(&new_state).unwrap();
    let msg = Message::Text(state_json);
    
    // Send to all connected clients
    for (_, tx) in state.users.read().await.iter() {
        // Non-blocking send, ignore errors (they'll be cleaned up when the connection drops)
        let _ = tx.try_send(msg.clone());
    }
}

// Initialize the Axum server
#[tokio::main]
async fn main() {
    // Initialize state
    let state = Arc::new(AppState {
        users: RwLock::new(HashMap::new()),
        latest_state: RwLock::new(SimulationState {
            timestamp: 0,
            traffic_data: vec![],
            energy_data: vec![],
            waste_data: vec![],
        }),
    });
    
    // Create a channel for simulation updates
    let (update_tx, mut update_rx) = mpsc::channel::<SimulationState>(100);
    
    // Clone state for the update handler
    let update_state = state.clone();
    
    // Spawn a task to handle simulation updates
    tokio::spawn(async move {
        while let Some(new_state) = update_rx.recv().await {
            update_simulation_state(update_state.clone(), new_state).await;
        }
    });
    
    // Build the router
    let app = Router::new()
        .route("/ws", get(websocket_handler))
        .layer(CorsLayer::permissive())
        .with_state(state);
    
    // Start the server
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3001").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
