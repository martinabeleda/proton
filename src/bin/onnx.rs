use onnxruntime::{environment::Environment, ndarray::Array, GraphOptimizationLevel, LoggingLevel};
use onnxruntime::tensor::OrtOwnedTensor;
use lazy_static::{lazy_static, __Deref};
use tracing::Level;
use tracing_subscriber::FmtSubscriber;
use std::sync::Arc;

lazy_static! {
    pub static ref ENVIRONMENT: Arc<Environment> = Arc::new(
        Environment::builder()
            .with_name("onnx_environment")
            .with_log_level(LoggingLevel::Warning)
            .build()
            .unwrap()
    );
}

fn main() {
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::TRACE)
        .finish();

    tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed");

    let mut session = ENVIRONMENT
        .new_session_builder()
        .unwrap()
        .with_optimization_level(GraphOptimizationLevel::Basic)
        .unwrap()
        .with_number_threads(1)
        .unwrap()
        .with_model_from_file("squeezenet1.0-8.onnx")
        .unwrap();

    let input0_shape: Vec<usize> = session.inputs[0]
        .dimensions()
        .map(std::option::Option::unwrap)
        .collect();
    let output0_shape: Vec<usize> = session.outputs[0]
        .dimensions()
        .map(std::option::Option::unwrap)
        .collect();

    assert_eq!(input0_shape, [1, 3, 224, 224]);
    assert_eq!(output0_shape, [1, 1000, 1, 1]);

    // initialize input data with values in [0.0, 1.0]
    let n: u32 = session.inputs[0]
        .dimensions
        .iter()
        .map(|d| d.unwrap())
        .product();

    let array = Array::linspace(0.0_f32, 1.0, n as usize)
        .into_shape(input0_shape)
        .unwrap();

    let input_tensor_values = vec![array.into()];

    let outputs: Vec<OrtOwnedTensor<f32, _>> = session.run(input_tensor_values).unwrap();

    println!("output={:?}", &outputs[0].deref());

}
