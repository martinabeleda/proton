pub mod backend;

use axum::extract::Extension;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::post;
use axum::{Json, Router};
use serde::Deserialize;
use std::collections::HashMap;
use std::error::Error;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::fs;

use crate::backend::onnx::infer_onnx_model;

#[derive(Debug, Deserialize)]
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
    println!("Inference: {:?}", models);

    let output = infer_onnx_model().await;

    (StatusCode::OK, "Hello World!")
}

#[tokio::main]
async fn main() {
    let config = load_config().await.unwrap();
    println!("Loaded config: {:?}", config);

    let models: HashMap<String, String> = config
        .models
        .into_iter()
        .map(|model_config| (model_config.name, model_config.path))
        .collect();

    let models = Arc::new(models);

    let app = Router::new()
        .route("/infer", post(handle_inference))
        .layer(Extension(models));

    let addr = SocketAddr::from(([127, 0, 0, 1], 8080));

    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await
        .unwrap();
}
