use ndarray::Array4;
use onnxruntime::environment::Environment;
use onnxruntime::session::Session;
use onnxruntime::tensor::OrtOwnedTensor;
use onnxruntime::{GraphOptimizationLevel, LoggingLevel};
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};

use crate::config::ModelConfig;

lazy_static! {
    pub static ref ENVIRONMENT: Arc<Environment> = Arc::new(
        Environment::builder()
            .with_name("onnx_environment")
            .with_log_level(LoggingLevel::Warning)
            .build()
            .unwrap()
    );
}

#[derive(Debug)]
pub struct Message {
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
/// # Example
///
/// ```
/// let model_config = &config.models[0];
/// let worker = InferenceWorker::new(model_config.name.clone(), model_config.path.clone()).await;
///
/// let (requests_tx, requests_rx) = mpsc::channel::<ChannelInferenceRequest>(32);
///
/// tokio::spawn(worker.run(requests_rx));
/// ```
///
impl InferenceWorker {
    pub fn new(config: &ModelConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }

    pub async fn run(&self, mut requests_rx: mpsc::Receiver<Message>) {
        let mut session = self.init_session();

        // Run the worker loop
        loop {
            let request = requests_rx.recv().await.unwrap();
            tracing::debug!("InferenceWorker received request {:?}", request);

            let outputs = session.run(vec![request.input_data]).unwrap();
            let output: &OrtOwnedTensor<f32, _> = &outputs[0];
            tracing::info!("Model output: {:?}", output);

            let probabilities: Vec<f32> = output
                .softmax(ndarray::Axis(1))
                .iter()
                .copied()
                .collect::<Vec<f32>>();

            // Send the prediction back to the handler
            let _ = request.response_tx.send(probabilities);
        }
    }

    fn init_session(&self) -> Session {
        // Load the onnx model and create a session. For now, we'll just
        // load the first model in the config but in the future, we'll want
        // the ability to load multiple models.
        let session = ENVIRONMENT
            .new_session_builder()
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Basic)
            .unwrap()
            .with_number_threads(1)
            .unwrap()
            .with_model_from_file(&self.config.path)
            .unwrap();

        tracing::info!("Created onnx session {:?}", &session);

        let input0_shape: Vec<usize> = session.inputs[0]
            .dimensions()
            .map(std::option::Option::unwrap)
            .collect();
        let output0_shape: Vec<usize> = session.outputs[0]
            .dimensions()
            .map(std::option::Option::unwrap)
            .collect();

        tracing::info!(
            "InferenceWorker created model with input shape: {:?} and output shape {:?}",
            input0_shape,
            output0_shape
        );

        session
    }
}
