use axum::extract::Extension;
use axum::response::IntoResponse;
use axum::Json;
use ndarray::Array4;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, oneshot};
use uuid::Uuid;

use crate::worker::Message;

#[derive(Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    pub model_name: String,
    pub data: Array4<f32>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct InferenceResponse {
    pub prediction_id: Uuid,
    pub data: Vec<f32>,
}

pub async fn handle_inference(
    Extension(requests_tx): Extension<Arc<mpsc::Sender<Message>>>,
    Json(request): Json<InferenceRequest>,
) -> impl IntoResponse {
    // Create a channel to receive the inference result
    let (response_tx, response_rx) = oneshot::channel();

    // Generate a prediction_id for this request
    let prediction_id = Uuid::new_v4();

    let message = Message {
        prediction_id,
        model_name: request.model_name,
        input_data: request.data,
        response_tx,
    };

    requests_tx
        .send_timeout(message, Duration::new(5, 0))
        .await
        .unwrap();
    let response = response_rx.await.unwrap();

    tracing::info!("Handler received prediction_id={:?}", prediction_id);

    Json(InferenceResponse {
        prediction_id,
        data: response,
    })
}
