use proton::config::Config;
use std::collections::HashMap;
use std::sync::Arc;
use std::thread;
use tokio::sync::mpsc::{channel, Sender};

use proton::logging;
use proton::server::{axum, grpc};
use proton::state::SharedState;
use proton::worker::{InferenceWorker, Message};

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

    // Spawn threads to run our workers in the background. We communicate with the threads using
    // the channels we created earlier. Each thread gets the receiving end and the sender sides
    // are provided to the inference handlers
    let mut queues_tx: HashMap<String, Sender<Message>> = HashMap::new();
    for model_config in config.models.iter() {
        // Create a separate queue for each worker so that worker can process messages at
        // different rates
        let (requests_tx, requests_rx) = channel::<Message>(config.server.buffer_size);
        queues_tx.insert(model_config.name.clone(), requests_tx);

        let mut worker = InferenceWorker::new(model_config.clone(), shared_state.clone());

        thread::spawn(move || {
            worker.run(requests_rx);
        });
    }
    let queues_tx = Arc::new(queues_tx);

    // Run both servers concurrently
    tokio::select! {
        grpc_result = axum::build(config.server.port, Arc::clone(&queues_tx), shared_state) => {
            tracing::info!("gRPC server exited");
            grpc_result.unwrap();
        }
        axum_result = grpc::build(config.server.grpc_port, queues_tx) => {
            tracing::info!("Axum server exited");
            axum_result.unwrap();
        }
    }
}
