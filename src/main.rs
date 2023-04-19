use axum::extract::Extension;
use axum::routing::{get, post};
use axum::Router;
use core::str::FromStr;
use proton::config::Config;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::thread;
use tokio::sync::mpsc;
use tracing::Level;

use proton::routes::{models, predict, ready};
use proton::state::SharedState;
use proton::worker::{InferenceWorker, Message};

#[tokio::main]
async fn main() {
    // Load config from file
    let config = Config::load("config.yaml").await.unwrap();
    let log_level = Level::from_str(&config.log_level).unwrap();

    // Set up loggging
    tracing_subscriber::fmt()
        .with_max_level(log_level)
        .with_file(true)
        .with_line_number(true)
        .init();

    tracing::info!("Loaded config: {:#?}", &config);

    let shared_state = Arc::new(SharedState::new(config.clone()));
    tracing::debug!("Shared state: {:?}", &shared_state);

    // Create separate queues for each model so that models can process messages at
    // different rates. We create a hash map keyed by model name
    let mut queues_tx: HashMap<String, mpsc::Sender<Message>> = HashMap::new();
    let mut queues_rx: HashMap<String, mpsc::Receiver<Message>> = HashMap::new();
    for model_config in config.models.iter() {
        let (tx, rx) = mpsc::channel::<Message>(config.server.buffer_size);
        queues_tx.insert(model_config.name.clone(), tx);
        queues_rx.insert(model_config.name.clone(), rx);
    }

    // Spawn threads to run our workers in the background. We communicate with the threads using
    // the channels we created earlier. Each thread gets the receiving end and the sender sides
    // are provided to the `handle_inference` function
    for model_config in config.models.iter() {
        let model_name = model_config.name.clone();
        tracing::info!("Creating InferenceWorker for {:?}", model_name);

        let requests_rx = queues_rx.remove(&model_name).unwrap();
        let mut worker = InferenceWorker::new(model_config.clone(), shared_state.clone());

        thread::spawn(move || {
            worker.run(requests_rx);
        });
    }

    let app = Router::new()
        .route("/predict", post(predict::handle_inference))
        .route("/models", get(models::get_models))
        .route("/ready", get(ready::get_health))
        .layer(Extension(shared_state))
        .layer(Extension(Arc::new(queues_tx)));

    let addr = SocketAddr::from(([127, 0, 0, 1], config.server.port));
    tracing::info!("Starting server, binding to port {:?}", &config.server.port);

    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await
        .unwrap();
}