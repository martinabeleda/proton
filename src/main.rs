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

#[derive(Debug, Deserialize)]
struct Config {
    models: Vec<ModelConfig>,
}

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
    Extension(models): Extension<Arc<HashMap<String, String>>>,
    Json(info): Json<InferenceRequest>,
) -> impl IntoResponse {
    let model_name = &info.model_name;
    let input_data = &info.input_data;

    println!("Inference for model: {:?}", model_name);
    println!("Input data: {:?}", input_data);
    println!("Models: {:?}", models);

    let _output = infer_onnx_model().await;

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

    // Load the onnx session. For now, we'll just load the first model in the config
    // but in the future, we'll want the ability to load multiple models
    let _session = &ENVIRONMENT
        .new_session_builder()
        .unwrap()
        .with_optimization_level(GraphOptimizationLevel::Basic)
        .unwrap()
        .with_number_threads(1)
        .unwrap()
        .with_model_from_file(&config.models[0].path)
        .unwrap();

    let app = Router::new()
        .route("/infer", post(handle_inference))
        .layer(Extension(models));

    let addr = SocketAddr::from(([127, 0, 0, 1], 8080));

    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await
        .unwrap();
}
