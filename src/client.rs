use ndarray::{Array, Array4};
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Serialize, Deserialize, Debug)]
struct InferenceRequest {
    model_name: String,
    data: Array4<f32>,
}

#[derive(Serialize, Deserialize, Debug)]
struct InferenceResponse {
    prediction_id: Uuid,
    data: Vec<f32>,
}

#[tokio::main]
async fn main() {
    // Set up loggging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_file(true)
        .with_line_number(true)
        .init();

    let input_shape = (1, 3, 224, 224);

    // Initialize input data with values in [0.0, 1.0]
    let n = input_shape.0 * input_shape.1 * input_shape.2 * input_shape.3;
    let array = Array::linspace(0.0_f32, 1.0, n as usize)
        .into_shape(input_shape)
        .unwrap();

    tracing::info!("Input array: {:?}", array);

    let model_name = "squeezenet";
    let data = InferenceRequest {
        model_name: model_name.to_string(),
        data: array,
    };

    let client = Client::new();
    let response = client
        .post("http://localhost:8080/predict")
        .json(&data)
        .send()
        .await
        .unwrap();

    match response.status() {
        StatusCode::OK => {
            match response.json::<InferenceResponse>().await {
                Ok(parsed) => tracing::info!("Success {:?}", parsed),
                Err(_) => tracing::error!("The response didn't match InferenceResponse"),
            };
        }
        _ => {
            tracing::error!("Something went wrong {:?}", response);
        }
    }
}
