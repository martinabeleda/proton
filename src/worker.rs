use ndarray::{Array, IxDyn};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};
use uuid::Uuid;

use crate::config::ModelConfig;
use crate::model::Model;
use crate::state::SharedState;

#[derive(Debug)]
pub struct Message {
    pub prediction_id: Uuid,
    pub model_name: String,
    pub input_data: Array<f32, IxDyn>,
    pub response_tx: oneshot::Sender<Array<f32, IxDyn>>,
}

pub struct InferenceWorker {
    pub config: ModelConfig,
    shared_state: Arc<SharedState>,
}

/// `InferenceWorker` is responsible for running inference on a specific ONNX model.
///
/// The worker loads the ONNX model and creates an ONNX session using the given model
/// name and path. It listens for incoming inference requests on a channel, and when
/// a request is received, the worker checks if the requested model name matches its
/// own. If so, it runs the inference and sends the result back to the request sender
/// through a one-shot channel.
///
impl InferenceWorker {
    pub fn new(config: ModelConfig, shared_state: Arc<SharedState>) -> Self {
        Self {
            config,
            shared_state,
        }
    }

    pub fn run(&mut self, mut requests_rx: mpsc::Receiver<Message>) {
        let mut model = Model::new(&self.config);
        tracing::info!("{:?} model ready", &self.config.name);

        // Update shared state to flag this model as ready
        self.shared_state
            .ready
            .get(&self.config.name)
            .unwrap()
            .store(true, Ordering::Relaxed);

        // Run the worker loop
        loop {
            let request = requests_rx.blocking_recv().unwrap();
            let model_name = request.model_name;
            let id = request.prediction_id;

            tracing::info!("{:?} got prediction_id={:?}", model_name, id);

            let output = model.predict(vec![request.input_data]);

            // Send the prediction back to the handler
            request.response_tx.send(output).unwrap();
            tracing::info!("{:?} sent prediction_id={:?}", model_name, id);
        }
    }
}
