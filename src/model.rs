use ndarray::Array4;
use onnxruntime::environment::Environment;
use onnxruntime::session::Session;
use onnxruntime::tensor::OrtOwnedTensor;
use onnxruntime::{GraphOptimizationLevel, LoggingLevel};
use std::sync::Arc;

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

pub struct Model<'a> {
    pub config: ModelConfig,
    session: Session<'a>,
}

impl Model<'_> {
    pub fn new(config: &ModelConfig) -> Self {
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
            .with_model_from_file(config.path.clone())
            .unwrap();

        tracing::info!(
            "Created onnx session for model: {:?} at path: {:?}",
            config.name,
            config.path
        );
        tracing::info!("Inputs: {:?}", session.inputs);
        tracing::info!("Outputs: {:?}", session.outputs);

        Self {
            config: config.clone(),
            session,
        }
    }

    pub fn predict(&mut self, inputs: Vec<Array4<f32>>) -> Vec<f32> {
        let outputs = self.session.run(inputs).unwrap();
        let output: &OrtOwnedTensor<f32, _> = &outputs[0];

        output
            .softmax(ndarray::Axis(1))
            .iter()
            .copied()
            .collect::<Vec<f32>>()
    }
}
