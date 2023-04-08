pub mod backend;

use axum::{
    routing::post,
    http::StatusCode,
    response::IntoResponse,
    Router, Json
};
use serde::Deserialize;
use std::net::SocketAddr;

use crate::backend::onnx::infer_onnx_model;

#[derive(Deserialize)]
struct InferenceRequest {
    model_type: String,
    model_path: String,
    input_data: Vec<f32>,
}

async fn handle_inference(Json(info): Json<InferenceRequest>) -> impl IntoResponse {
    let output = infer_onnx_model().await;

    (StatusCode::OK, "Hello World!")
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
