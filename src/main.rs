pub mod backend;

use axum::extract::Extension;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::post;
use axum::{Json, Router};
use onnxruntime::environment::Environment;
use onnxruntime::{GraphOptimizationLevel, LoggingLevel};
use serde::Deserialize;
use std::collections::HashMap;
use std::error::Error;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::fs;
use tokio::sync::{mpsc, oneshot};

use crate::backend::onnx::infer_onnx_model;

#[derive(Clone, Debug, Deserialize)]
struct ModelConfig {
    name: String,
    path: String,
}

#[derive(Clone, Debug, Deserialize)]
struct Config {
    models: Vec<ModelConfig>,
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
    async fn new(config: &ModelConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }

    async fn run(&self, mut requests_rx: mpsc::Receiver<InferenceMessage>) {

        let environment = Environment::builder()
            .with_name("onnx proton")
            .with_log_level(LoggingLevel::Warning)
            .build()
            .unwrap();

        // Load the onnx model and create a session. For now, we'll just 
        // load the first model in the config but in the future, we'll want 
        // the ability to load multiple models.
        let _session = &environment
            .new_session_builder()
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Basic)
            .unwrap()
            .with_number_threads(1)
            .unwrap()
            .with_model_from_file(&self.config.path)
            .unwrap();

        // Run the worker loop
        loop {
            if let Some(request) = requests_rx.recv().await {
                // Run inference using the onnx session
                let result = infer_onnx_model().await;

                // Send the prediction back to the handler
                let _ = request.response_tx.send(result);
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
    Extension(models): Extension<Arc<HashMap<String, String>>>,
    Json(request): Json<InferenceRequest>,
) -> impl IntoResponse {
    println!("Inference for model: {:?}", &request.model_name);
    println!("Input data: {:?}", &request.input_data);
    println!("Models: {:?}", models);

    // Create a channel to receive the inference result
    let (response_tx, response_rx) = oneshot::channel();

    let request = InferenceMessage {
        model_name: request.model_name,
        input_data: request.input_data,
        response_tx,
    };

    if let Err(err) = requests_tx.send(request).await {
        panic!("Error sending request to worker: {:?}", err);
    }

    let response = response_rx.await;

    println!("Handler received prediction: {:?}", response);

    (StatusCode::OK, "Hello World!")
}

#[tokio::main]
async fn main() {
    let config = load_config().await.unwrap();
    println!("Loaded config: {:?}", config);

    let models: HashMap<String, String> = config
        .models
        .clone()
        .into_iter()
        .map(|model_config| (model_config.name, model_config.path))
        .collect();

    let models = Arc::new(models);

    // Create a channel for the inference worker to listen for inference
    // requests on
    let (requests_tx, requests_rx) = mpsc::channel::<InferenceMessage>(32);


    let worker_task = tokio::spawn(async move {
        let worker = InferenceWorker::new(&config.models[0].clone()).await;
        worker.run(requests_rx).await;
    });

    let app = Router::new()
        .route("/predict", post(handle_inference))
        .layer(Extension(Arc::new(requests_tx)))
        .layer(Extension(models));

    let addr = SocketAddr::from(([127, 0, 0, 1], 8080));

    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await
        .unwrap();

    worker_task.await.unwrap();
}
