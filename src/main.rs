#[macro_use]
extern crate lazy_static;

use crate::config::{Config, ServerConfig};
use axum::extract::Extension;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::post;
use axum::{Json, Router};
use serde::Deserialize;
use std::error::Error;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::fs;
use tokio::sync::{mpsc, oneshot};
use tokio::task::LocalSet;

pub mod backend;
pub mod config;
pub mod worker;

#[derive(Deserialize)]
struct InferenceRequest {
    model_name: String,
    input_data: Vec<f32>,
}

async fn load_config() -> Result<Config, Box<dyn Error>> {
    let config_data = fs::read_to_string("config.yaml").await?;
    let config: Config = serde_yaml::from_str(&config_data)?;
    Ok(config)
}

async fn handle_inference(
    Extension(requests_tx): Extension<Arc<mpsc::Sender<worker::Message>>>,
    Json(request): Json<InferenceRequest>,
) -> impl IntoResponse {
    tracing::info!("Inference for model: {:?}", &request.model_name);
    tracing::info!("Input data: {:?}", &request.input_data);

    // Create a channel to receive the inference result
    let (response_tx, response_rx) = oneshot::channel();

    let request = worker::Message {
        model_name: request.model_name,
        input_data: request.input_data,
        response_tx,
    };

    tracing::info!("Handler sending message: {:#?}", request);

    if let Err(err) = requests_tx.send(request).await {
        panic!("Error sending request to worker: {:?}", err);
    }

    let response = response_rx.await.unwrap();

    tracing::info!("Handler received prediction: {:?}", response);

    (StatusCode::OK, "Hello World!")
}

async fn run_server(config: ServerConfig, requests_tx: mpsc::Sender<worker::Message>) {
    tracing::info!("Starting server");
    let app = Router::new()
        .route("/predict", post(handle_inference))
        .layer(Extension(Arc::new(requests_tx)));

    let addr = SocketAddr::from(([127, 0, 0, 1], config.port));

    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await
        .unwrap();
    tracing::info!("Server started, bound to port {:?}", config.port);
}

#[tokio::main]
async fn main() {
    // Set up loggging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_file(true)
        .with_line_number(true)
        .init();

    let config = load_config().await.unwrap();
    tracing::info!("Loaded config: {:#?}", config);

    // Create a channel for the inference worker to listen for inference
    // requests on
    let (requests_tx, requests_rx) = mpsc::channel::<worker::Message>(config.server.buffer_size);

    tracing::info!("Creating InferenceWorker");
    let inference_worker = Arc::new(worker::InferenceWorker::new(&config.models[0]));

    // Spawn the inference worker task in the current thread since it is not Send
    let local = LocalSet::new();
    local.spawn_local(async move {
        inference_worker.run(requests_rx).await;
    });

    let server_task = run_server(config.server, requests_tx);

    // Run the worker and the server concurrently
    tokio::join!(local, server_task);
}
