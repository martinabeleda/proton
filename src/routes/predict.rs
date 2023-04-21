use axum::extract::Extension;
use axum::response::IntoResponse;
use axum::Json;
use ndarray::{Array, IxDyn};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};
use uuid::Uuid;

use crate::worker::Message;

#[derive(Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    pub model_name: String,
    pub data: Array<f32, IxDyn>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct InferenceResponse {
    pub prediction_id: Uuid,
    pub model_name: String,
    pub data: Array<f32, IxDyn>,
}

pub async fn handle_inference(
    Extension(queues_tx): Extension<Arc<HashMap<String, mpsc::Sender<Message>>>>,
    Json(request): Json<InferenceRequest>,
) -> impl IntoResponse {
    let prediction_id = Uuid::new_v4();

    // Create a channel to receive the inference result
    let (tx, rx) = oneshot::channel();

    let message = Message {
        prediction_id,
        model_name: request.model_name.clone(),
        input_data: request.data,
        response_tx: tx,
    };

    queues_tx
        .get(&request.model_name)
        .unwrap()
        .send(message)
        .await
        .unwrap();

    let response = rx.await.unwrap();

    tracing::info!(
        "Handler received prediction_id={:?} for model={}",
        prediction_id,
        &request.model_name
    );

    Json(InferenceResponse {
        prediction_id,
        model_name: request.model_name.clone(),
        data: response,
    })
}
