/// Onnx test script
///
/// This script is used to test running an onnx session with a variety of
/// models.
#[allow(dead_code, unused_variables)]
use lazy_static::lazy_static;
use ndarray::{Array, Dimension, IxDyn};
use onnxruntime::tensor::OrtOwnedTensor;
use onnxruntime::{environment::Environment, GraphOptimizationLevel, LoggingLevel};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::Level;
use tracing_subscriber::FmtSubscriber;

lazy_static! {
    pub static ref ENVIRONMENT: Arc<Environment> = Arc::new(
        Environment::builder()
            .with_name("onnx_environment")
            .with_log_level(LoggingLevel::Warning)
            .build()
            .unwrap()
    );
}

#[derive(Debug, Deserialize, Serialize)]
struct Output<A> {
    data: Array<A, IxDyn>,
}

trait Model {
    fn dummy_data(&self) -> Array<f32, IxDyn>;
}

struct Squeezenet {
    name: String,
    file: String,
}

impl Squeezenet {
    fn new() -> Self {
        Squeezenet {
            name: "squeezenet".to_string(),
            file: "squeezenet1.0-8.onnx".to_string(),
        }
    }
}

impl Model for Squeezenet {
    fn dummy_data(&self) -> Array<f32, IxDyn> {
        let input_shape = IxDyn(&[1, 3, 224, 224]);

        Array::linspace(0.0_f32, 1.0, input_shape.size())
            .into_shape(input_shape)
            .unwrap()
    }
}

struct MaskRCNN {
    name: String,
    file: String,
}

impl MaskRCNN {
    fn new() -> Self {
        MaskRCNN {
            name: "maskrcnn".to_string(),
            file: "MaskRCNN-10.onnx".to_string(),
        }
    }
}

impl Model for MaskRCNN {
    fn dummy_data(&self) -> Array<f32, IxDyn> {
        let input_shape = IxDyn(&[3, 224, 224]);

        Array::linspace(0.0_f32, 1.0, input_shape.size())
            .into_shape(input_shape)
            .unwrap()
    }
}

fn main() {
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();

    tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed");

    let model = MaskRCNN::new();
    println!("Created model {}", model.name);

    let mut session = ENVIRONMENT
        .new_session_builder()
        .unwrap()
        .with_optimization_level(GraphOptimizationLevel::Basic)
        .unwrap()
        .with_number_threads(1)
        .unwrap()
        .with_model_from_file(&model.file)
        .unwrap();

    println!("Session inputs: {:?}", session.inputs);
    println!("Session outputs: {:?}", session.outputs);

    let input_tensor_values = vec![model.dummy_data().into()];

    let outputs: Vec<OrtOwnedTensor<f32, _>> = session.run(input_tensor_values).unwrap();

    let output_data =
        Array::from_shape_vec(outputs[0].shape(), outputs[0].iter().cloned().collect()).unwrap();
    let output = Output { data: output_data };

    println!("output={:?}", output);

    println!("json={:?}", serde_json::to_string(&output).unwrap());
}
