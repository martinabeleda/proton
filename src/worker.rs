use onnxruntime::environment::Environment;
use onnxruntime::{GraphOptimizationLevel, LoggingLevel};
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};

use crate::backend::onnx;
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
    pub input_data: Vec<f32>,
    pub response_tx: oneshot::Sender<Option<Vec<f32>>>,
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
        // Load the onnx model and create a session. For now, we'll just
        // load the first model in the config but in the future, we'll want
        // the ability to load multiple models.
        let session = &ENVIRONMENT
            .new_session_builder()
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Basic)
            .unwrap()
            .with_number_threads(1)
            .unwrap()
            .with_model_from_file(&self.config.path)
            .unwrap();

        tracing::info!("Created onnx session {:?}", session);

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

        // Run the worker loop
        loop {
            if let Some(request) = requests_rx.recv().await {
                tracing::info!("InferenceWorker received request {:?}", request);
                // Run inference using the onnx session
                let result = onnx::infer_onnx_model();

                // Send the prediction back to the handler
                let _ = request.response_tx.send(result);
            } else {
                tracing::info!("InferenceWorker error receiving request");
            }
        }
    }
}
