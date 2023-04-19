use axum::extract::Extension;
use axum::response::IntoResponse;
use axum::Json;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::config::Config;

#[derive(Clone, Serialize, Deserialize)]
pub struct ModelsResponse {
    pub models: Vec<Model>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Model {
    pub name: String,
}

pub async fn get_models(Extension(config): Extension<Arc<Config>>) -> impl IntoResponse {
    let models: Vec<Model> = config
        .models
        .iter()
        .map(|model_config| Model {
            name: model_config.name.clone(),
        })
        .collect();

    Json(ModelsResponse { models })
}
