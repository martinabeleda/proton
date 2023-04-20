use axum::extract::Extension;
use axum::routing::{get, post};
use axum::Router;
use axum_prometheus::PrometheusMetricLayer;
use proton::config::Config;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::thread;
use tokio::sync::mpsc;
use tonic::transport::Server as GrpcServer;
use tonic::{Request, Response, Status};

use proton::grpc::predictor_server::{Predictor, PredictorServer};
use proton::grpc::{InferenceRequest, InferenceResponse};
use proton::logging;
use proton::routes::{models, predict, ready};
use proton::state::SharedState;
use proton::worker::{InferenceWorker, Message};

#[derive(Debug, Default)]
pub struct MyPredictService;

#[tonic::async_trait]
impl Predictor for MyPredictService {
    async fn predict(
        &self,
        request: Request<InferenceRequest>,
    ) -> Result<Response<InferenceResponse>, Status> {
        let _input = request.into_inner();

        let response = InferenceResponse {
            model_name: String::from("my_model"),
            prediction_id: String::from("some_id"),
            data: vec![0.0, 1.0],
            shape: vec![2],
        };

        Ok(Response::new(response))
    }
}

#[tokio::main]
async fn main() {
    let config = match Config::load("config.yaml") {
        Ok(config) => config,
        Err(err) => panic!("Failed to load config {:?}", err),
    };

    logging::setup(&config.log_level);

    tracing::info!("Starting proton with config: {:#?}", &config);

    // Create shared state behine atomic referenced counter to
    // store config and model readiness state
    let shared_state = Arc::new(SharedState::new(config.clone()));
    tracing::debug!("Shared state: {:?}", &shared_state);

    // Spawn threads to run our workers in the background. We communicate with the threads using
    // the channels we created earlier. Each thread gets the receiving end and the sender sides
    // are provided to the `handle_inference` function
    let mut queues_tx: HashMap<String, mpsc::Sender<Message>> = HashMap::new();
    for model_config in config.models.iter() {
        // Create a separate queue for each worker so that worker can process messages at
        // different rates
        let (requests_tx, requests_rx) = mpsc::channel::<Message>(config.server.buffer_size);
        queues_tx.insert(model_config.name.clone(), requests_tx);

        let mut worker = InferenceWorker::new(model_config.clone(), shared_state.clone());

        thread::spawn(move || {
            worker.run(requests_rx);
        });
    }

    let (prometheus_layer, metric_handle) = PrometheusMetricLayer::pair();

    let app = Router::new()
        .route("/predict", post(predict::handle_inference))
        .route("/models", get(models::get_models))
        .route("/ready", get(ready::get_health))
        .route("/metrics", get(|| async move { metric_handle.render() }))
        .layer(prometheus_layer)
        .layer(Extension(shared_state))
        .layer(Extension(Arc::new(queues_tx)));

    let addr = SocketAddr::from(([0, 0, 0, 0], config.server.port));
    tracing::info!("Starting server, binding to port {:?}", &config.server.port);

    let axum_server = axum::Server::bind(&addr).serve(app.into_make_service());

    let grpc_addr = SocketAddr::from(([0, 0, 0, 0], config.server.grpc_port));
    let grpc_server = GrpcServer::builder()
        .add_service(PredictorServer::new(MyPredictService {}))
        .serve(grpc_addr);

    tracing::info!("Starting servers");

    // Run both servers concurrently
    tokio::select! {
        axum_result = axum_server => {
            tracing::info!("Axum server exited");
            axum_result.unwrap();
        }
        grpc_result = grpc_server => {
            tracing::info!("gRPC server exited");
            grpc_result.unwrap();
        }
    }
}
