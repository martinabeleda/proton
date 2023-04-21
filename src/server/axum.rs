use axum::extract::Extension;
use axum::routing::{get, post};
use axum::{Router, Server};
use axum_prometheus::PrometheusMetricLayer;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::mpsc::Sender;

use crate::routes::{models, predict, ready};
use crate::state::SharedState;
use crate::worker::Message;

pub async fn build(
    port: u16,
    queues_tx: HashMap<String, Sender<Message>>,
    shared_state: Arc<SharedState>,
) -> Result<(), hyper::Error> {
    let (prometheus_layer, metric_handle) = PrometheusMetricLayer::pair();

    let app = Router::new()
        .route("/predict", post(predict::handle_inference))
        .route("/models", get(models::get_models))
        .route("/ready", get(ready::get_health))
        .route("/metrics", get(|| async move { metric_handle.render() }))
        .layer(prometheus_layer)
        .layer(Extension(shared_state))
        .layer(Extension(Arc::new(queues_tx)));

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    tracing::info!("Starting server, binding to port {:?}", port);

    Server::bind(&addr).serve(app.into_make_service()).await
}
