use ndarray::{Array, Dimension, IxDyn};
use proton::routes::predict::{InferenceRequest, InferenceResponse};
use rand::prelude::*;
use reqwest::{Client, StatusCode};
use std::collections::HashMap;
use std::time::Duration;
use tokio::task::JoinHandle;

trait Model {
    fn dummy_data(&self) -> Array<f32, IxDyn>;

    fn get_name(&self) -> String;
}

struct Squeezenet {
    name: String,
}

impl Squeezenet {
    fn new() -> Self {
        Squeezenet {
            name: "squeezenet".to_string(),
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

    fn get_name(&self) -> String {
        self.name.clone()
    }
}

struct MaskRCNN {
    name: String,
}

impl MaskRCNN {
    fn new() -> Self {
        MaskRCNN {
            name: "maskrcnn".to_string(),
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

    fn get_name(&self) -> String {
        self.name.clone()
    }
}

async fn send_request(client: &Client, data: &InferenceRequest) -> (String, Duration) {
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
                Ok(parsed) => tracing::info!(
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

    (data.model_name.clone(), start_time.elapsed())
}

#[tokio::main]
async fn main() {
    // Set up loggging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_file(true)
        .with_line_number(true)
        .init();

    let mut models: Vec<Box<dyn Model>> = Vec::new();
    models.push(Box::new(MaskRCNN::new()));
    models.push(Box::new(Squeezenet::new()));

    let client = Client::new();

    let num_requests = 20; // number of concurrent requests
    let mut futures: Vec<JoinHandle<(String, Duration)>> = Vec::with_capacity(num_requests);
    let mut rng = thread_rng();

    for _ in 0..num_requests {
        let model = models.choose(&mut rng).unwrap();
        tracing::info!("Sending request for {:?}", &model.get_name());
        let data = InferenceRequest {
            model_name: model.get_name(),
            data: model.dummy_data(),
        };

        let client = client.clone();
        let task = tokio::spawn(async move { send_request(&client, &data).await });
        futures.push(task);
    }

    let mut elapsed_times: HashMap<String, Vec<Duration>> = HashMap::new();

    for future in futures {
        let (model_name, duration) = future.await.unwrap();
        elapsed_times
            .entry(model_name)
            .or_insert_with(Vec::new)
            .push(duration);
    }

    for (model_name, times) in &elapsed_times {
        let total_time: Duration = times.iter().cloned().sum();
        let average_time = total_time / times.len() as u32;
        let mut sorted_times = times.clone();
        sorted_times.sort_unstable();

        let p50 = sorted_times[((0.50 * (times.len() as f64)).round() as usize) - 1];
        let p95 = sorted_times[((0.95 * (times.len() as f64)).round() as usize) - 1];
        let p99 = sorted_times[((0.99 * (times.len() as f64)).round() as usize) - 1];

        tracing::info!("\nModel: {:?}", model_name);
        tracing::info!("Total time: {:?}", total_time);
        tracing::info!("Average time: {:?}", average_time);
        tracing::info!("Median time: {:?}", p50);
        tracing::info!("p95 time: {:?}", p95);
        tracing::info!("p99 time: {:?}", p99);
    }
}
