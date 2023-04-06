use onnxruntime::{environment::Environment, tensor::OrtOwnedTensor};
use std::error::Error;

async fn infer_onnx_model(
    model_path: &str,
    input_data: Vec<f32>,
) -> Result<Vec<f32>, Box<dyn Error>> {
    let environment = Environment::builder().build()?;
    let session = environment
        .new_session_builder()?
        .with_model_from_file(model_path)?;

    let input_shape = &[1, 224, 224, 3];
    let input_tensor = onnxruntime::Tensor::new_float(input_shape, input_data)?;
    let output: Vec<OrtOwnedTensor<f32, _>> = session.run(vec![&input_tensor])?;

    Ok(output[0].view::<f32>().unwrap().to_vec())
}

