use axum::extract::Extension;
use axum::response::IntoResponse;
use axum::Json;
use serde::{Deserialize, Serialize};
use std::sync::atomic::Ordering;
use std::sync::Arc;

use crate::state::SharedState;

#[derive(Clone, Serialize, Deserialize)]
pub struct ReadyResponse {
    pub healthy: bool,
}

pub async fn get_health(Extension(state): Extension<Arc<SharedState>>) -> impl IntoResponse {
    // Service is ready once all models have been initialized
    let healthy = state
        .ready
        .iter()
        .all(|(_, ready)| ready.load(Ordering::Relaxed));

    Json(ReadyResponse { healthy })
}
