use ndarray::Array4;
use tokio::sync::{mpsc, oneshot};
use uuid::Uuid;

use crate::config::ModelConfig;
use crate::model::Model;

#[derive(Debug)]
pub struct Message {
    pub prediction_id: Uuid,
    pub model_name: String,
    pub input_data: Array4<f32>,
    pub response_tx: oneshot::Sender<Vec<f32>>,
}

pub struct InferenceWorker {
    pub config: ModelConfig,
}

/// `InferenceWorker` is responsible for running inference on a specific ONNX model.
///
/// The worker loads the ONNX model and creates an ONNX session using the given model name and path.
/// It listens for incoming inference requests on a channel, and when a request is received, the worker
/// checks if the requested model name matches its own. If so, it runs the inference and sends the result
/// back to the request sender through a one-shot channel.
///
impl InferenceWorker {
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }

    pub fn run(&mut self, mut requests_rx: mpsc::Receiver<Message>) {
        let mut model = Model::new(&self.config);

        // Run the worker loop
        loop {
            let request = requests_rx.blocking_recv().unwrap();

            tracing::info!("Got prediction_id={:?}", request.prediction_id);

            let output = model.predict(vec![request.input_data]);

            // Send the prediction back to the handler
            let _ = request.response_tx.send(output);
            tracing::info!("Sent prediction_id={:?}", request.prediction_id);
        }
    }
}
