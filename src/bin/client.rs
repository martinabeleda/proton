use ndarray::Array;
use proton::predict::{InferenceRequest, InferenceResponse};
use rand::prelude::*;
use reqwest::{Client, StatusCode};
use std::time::Duration;
use tokio::task::JoinHandle;

async fn send_request(client: &Client, data: &InferenceRequest) -> Duration {
    let start_time = tokio::time::Instant::now();
    let response = client
        .post("http://localhost:8080/predict")
        .json(&data)
        .send()
        .await
        .unwrap();

    match response.status() {
        StatusCode::OK => {
            match response.json::<InferenceResponse>().await {
                Ok(parsed) => tracing::debug!(
                    "Success for model {:?}, prediction_id: {:?}",
                    parsed.model_name,
                    parsed.prediction_id
                ),
                Err(_) => tracing::error!("The response didn't match InferenceResponse"),
            };
        }
        _ => {
            tracing::error!("Something went wrong {:?}", response);
        }
    }

    start_time.elapsed()
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

    tracing::info!("Input array: {:?}", array.shape());

    let model_names = vec!["squeezenet", "squeezenet2"];

    let client = Client::new();

    let num_requests = 100; // number of concurrent requests
    let mut futures: Vec<JoinHandle<Duration>> = Vec::with_capacity(num_requests);
    let mut rng = thread_rng();

    let start_time = tokio::time::Instant::now();
    for _ in 0..num_requests {
        let model_name = model_names.choose(&mut rng).unwrap();
        tracing::debug!("Sending request for {:?}", model_name);
        let data = InferenceRequest {
            model_name: model_name.to_string(),
            data: array.clone(),
        };

        let client = client.clone();
        let task = tokio::spawn(async move { send_request(&client, &data).await });
        futures.push(task);
    }

    let mut elapsed_times = Vec::with_capacity(num_requests);
    for future in futures {
        elapsed_times.push(future.await.unwrap());
    }

    let total_time = start_time.elapsed();
    elapsed_times.sort_unstable();

    let sum = elapsed_times.iter().cloned().sum::<Duration>();
    let average = sum / num_requests as u32;

    tracing::info!("Elapsed times: {:?}", elapsed_times);

    let p95 = elapsed_times[((0.95 * (num_requests as f64)).round() as usize) - 1];
    let p99 = elapsed_times[((0.99 * (num_requests as f64)).round() as usize) - 1];

    tracing::info!("Total time: {:?}", total_time);
    tracing::info!("Average time: {:?}", average);
    tracing::info!("p95 time: {:?}", p95);
    tracing::info!("p99 time: {:?}", p99);
}