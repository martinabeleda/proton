use axum::{
    extract::{Extension, Json},
    handler::post,
    http::StatusCode,
    response::IntoResponse,
    Router,
};
use serde::Deserialize;
use std::net::SocketAddr;
use tower::ServiceBuilder;
use tower_http::trace::TraceLayer;

use crate::backend::onnx::infer_onnx_model;

#[derive(Deserialize)]
struct InferenceRequest {
    model_type: String,
    model_path: String,
    input_data: Vec<f32>,
}

async fn handle_inference(Json(info): Json<InferenceRequest>) -> impl IntoResponse {
    let model_type = &info.model_type;
    let model_path = &info.model_path;
    let input_data = &info.input_data;

    match model_type.as_str() {
        "onnx" => {
            let output = infer_onnx_model(model_path, input_data.to_owned()).await;
            // Handle the result and return the appropriate response
        }
        // Implement other model types similarly
        _ => (StatusCode::BAD_REQUEST, "Invalid model type"),
    }
}

// basic handler that responds with a static string
async fn root() -> &'static str {
    "Hello, World!"
}

#[tokio::main]
async fn main() {
    let app = Router::new()
        .route("/infer", post(handle_inference));

    let addr = SocketAddr::from(([127, 0, 0, 1], 8080));

    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await
        .unwrap();
}
