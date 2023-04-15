use axum::extract::Extension;
use axum::routing::post;
use axum::Router;
use proton::config::{Config, ServerConfig};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::sync::Mutex;
use tokio::task::LocalSet;

use proton::predict::handle_inference;
use proton::worker::{InferenceWorker, Message};

async fn run_server(config: ServerConfig, queues_tx: HashMap<String, mpsc::Sender<Message>>) {
    let app = Router::new()
        .route("/predict", post(handle_inference))
        .layer(Extension(Arc::new(queues_tx)));

    let addr = SocketAddr::from(([127, 0, 0, 1], config.port));
    tracing::info!("Starting server, binding to port {:?}", config.port);

    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await
        .unwrap();
}

#[tokio::main]
async fn main() {
    // Set up loggging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_file(true)
        .with_line_number(true)
        .init();

    let config = Config::load("config.yaml").await.unwrap();
    tracing::info!("Loaded config: {:#?}", config);

    // Create separate queues for each model so that models can process messages at
    // different rates. We create a hash map keyed by model name
    let mut queues_tx: HashMap<String, mpsc::Sender<Message>> = HashMap::new();
    let mut queues_rx: HashMap<String, mpsc::Receiver<Message>> = HashMap::new();
    for model_config in config.models.iter() {
        let (tx, rx) = mpsc::channel::<Message>(config.server.buffer_size);
        queues_tx.insert(model_config.name.clone(), tx);
        queues_rx.insert(model_config.name.clone(), rx);
    }

    let local = LocalSet::new();
    for model_config in config.models.iter() {
        let model_name = model_config.name.clone();
        tracing::info!("Creating InferenceWorker for {:?}", model_name);

        let requests_rx = queues_rx.remove(&model_name).unwrap();
        let inference_worker = Arc::new(Mutex::new(InferenceWorker::new(&model_config)));

        local.spawn_local(async move {
            let mut worker = inference_worker.lock().await;
            worker.run(requests_rx).await;
        });
    }

    let server_task = run_server(config.server, queues_tx);

    // Run the worker and the server concurrently
    tokio::join!(local, server_task);
}
