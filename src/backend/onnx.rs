use std::error::Error;

pub async fn infer_onnx_model() -> Result<Vec<f32>, Box<dyn Error>> {
    Ok(vec![0.1, 0.2, 0.3])
}

