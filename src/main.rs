pub mod backend;

use axum::extract::Extension;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::post;
use axum::{Json, Router};
use onnxruntime::environment::Environment;
use onnxruntime::{GraphOptimizationLevel, LoggingLevel};
use serde::Deserialize;
use std::error::Error;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::fs;
use tokio::sync::{mpsc, oneshot};
use tokio::task::LocalSet;

use crate::backend::onnx::infer_onnx_model;

#[macro_use]
extern crate lazy_static;

lazy_static! {
    pub static ref ENVIRONMENT: Arc<Environment> = Arc::new(
        Environment::builder()
            .with_name("onnx proton")
            .with_log_level(LoggingLevel::Warning)
            .build()
            .unwrap()
    );
}

#[derive(Clone, Debug, Deserialize)]
struct ModelConfig {
    name: String,
    path: String,
}

#[derive(Clone, Debug, Deserialize)]
struct ServerConfig {
    buffer_size: usize,
    port: u16,
}

#[derive(Clone, Debug, Deserialize)]
struct Config {
    models: Vec<ModelConfig>,
    server: ServerConfig,
}

#[derive(Deserialize)]
struct InferenceRequest {
    model_name: String,
    input_data: Vec<f32>,
}

#[derive(Debug)]
struct InferenceMessage {
    model_name: String,
    input_data: Vec<f32>,
    response_tx: oneshot::Sender<Option<Vec<f32>>>,
}

struct InferenceWorker {
    config: ModelConfig,
}

/// `InferenceWorker` is responsible for running inference on a specific ONNX model.
///
/// The worker loads the ONNX model and creates an ONNX session using the given model name and path.
/// It listens for incoming inference requests on a channel, and when a request is received, the worker
/// checks if the requested model name matches its own. If so, it runs the inference and sends the result
/// back to the request sender through a one-shot channel.
///
/// # Example
///
/// ```
/// let model_config = &config.models[0];
/// let worker = InferenceWorker::new(model_config.name.clone(), model_config.path.clone()).await;
///
/// let (requests_tx, requests_rx) = mpsc::channel::<ChannelInferenceRequest>(32);
///
/// tokio::spawn(worker.run(requests_rx));
/// ```
///
impl InferenceWorker {
    fn new(config: &ModelConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }

    async fn run(&self, mut requests_rx: mpsc::Receiver<InferenceMessage>) {
        // Load the onnx model and create a session. For now, we'll just
        // load the first model in the config but in the future, we'll want
        // the ability to load multiple models.
        let session = &ENVIRONMENT
            .new_session_builder()
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Basic)
            .unwrap()
            .with_number_threads(1)
            .unwrap()
            .with_model_from_file(&self.config.path)
            .unwrap();

        tracing::info!("Created onnx session {:?}", session);

        let input0_shape: Vec<usize> = session.inputs[0]
            .dimensions()
            .map(std::option::Option::unwrap)
            .collect();
        let output0_shape: Vec<usize> = session.outputs[0]
            .dimensions()
            .map(std::option::Option::unwrap)
            .collect();

        tracing::info!(
            "InferenceWorker created model with input shape: {:?} and output shape {:?}",
            input0_shape,
            output0_shape
        );

        // Run the worker loop
        loop {
            if let Some(request) = requests_rx.recv().await {
                tracing::info!("InferenceWorker received request {:?}", request);
                // Run inference using the onnx session
                let result = infer_onnx_model();

                // Send the prediction back to the handler
                let _ = request.response_tx.send(result);
            } else {
                tracing::info!("InferenceWorker error receiving request");
            }
        }
    }
}

async fn load_config() -> Result<Config, Box<dyn Error>> {
    let config_data = fs::read_to_string("config.yaml").await?;
    let config: Config = serde_yaml::from_str(&config_data)?;
    Ok(config)
}

async fn handle_inference(
    Extension(requests_tx): Extension<Arc<mpsc::Sender<InferenceMessage>>>,
    Json(request): Json<InferenceRequest>,
) -> impl IntoResponse {
    tracing::info!("Inference for model: {:?}", &request.model_name);
    tracing::info!("Input data: {:?}", &request.input_data);

    // Create a channel to receive the inference result
    let (response_tx, response_rx) = oneshot::channel();

    let request = InferenceMessage {
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

async fn run_server(config: ServerConfig, requests_tx: mpsc::Sender<InferenceMessage>) {
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
    let (requests_tx, requests_rx) = mpsc::channel::<InferenceMessage>(config.server.buffer_size);

    tracing::info!("Creating InferenceWorker");
    let worker = Arc::new(InferenceWorker::new(&config.models[0]));

    // Spawn the inference worker task in the current thread since it is not Send
    let local = LocalSet::new();
    local.spawn_local(async move {
        worker.run(requests_rx).await;
    });

    let server_task = run_server(config.server, requests_tx);

    // Run the worker and the server concurrently
    tokio::join!(local, server_task);
}
