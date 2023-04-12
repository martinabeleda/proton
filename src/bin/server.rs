use axum::extract::Extension;
use axum::routing::post;
use axum::Router;
use proton::config::{Config, ServerConfig};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::sync::Mutex;
use tokio::task::LocalSet;

use proton::predict::handle_inference;
use proton::worker::{InferenceWorker, Message};

async fn run_server(config: ServerConfig, requests_tx: mpsc::Sender<Message>) {
    let app = Router::new()
        .route("/predict", post(handle_inference))
        .layer(Extension(Arc::new(requests_tx)));

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

    // Create a channel for the inference worker to listen for inference
    // requests on
    let (requests_tx, requests_rx) = mpsc::channel::<Message>(config.server.buffer_size);

    tracing::info!("Creating InferenceWorker");
    let inference_worker = Arc::new(Mutex::new(InferenceWorker::new(&config)));

    // Spawn the inference worker task in the current thread since it is not Send
    let local = LocalSet::new();
    local.spawn_local(async move {
        let mut worker = inference_worker.lock().await;
        worker.run(requests_rx).await;
    });

    let server_task = run_server(config.server, requests_tx);

    // Run the worker and the server concurrently
    tokio::join!(local, server_task);
}
