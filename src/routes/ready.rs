use axum::extract::Extension;
use axum::response::IntoResponse;
use axum::Json;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::state::SharedState;

#[derive(Clone, Serialize, Deserialize)]
pub struct ReadyResponse {
    pub healthy: bool,
}

pub async fn get_health(Extension(state): Extension<Arc<SharedState>>) -> impl IntoResponse {
    let healthy = false;

    Json(ReadyResponse { healthy })
}
