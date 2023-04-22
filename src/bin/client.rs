use proton::routes::predict::{InferenceRequest, InferenceResponse};
use rand::prelude::*;
use reqwest::{Client, StatusCode};
use std::collections::HashMap;
use std::time::Duration;
use tokio::task::JoinHandle;

use proton::utils::{analyze_results, MaskRCNN, Model, Squeezenet};

const NUM_REQUESTS: usize = 20;

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
                Ok(parsed) => println!(
                    "Success for model {:?}, prediction_id: {:?}",
                    parsed.model_name, parsed.prediction_id
                ),
                Err(err) => panic!("The response didn't match InferenceResponse {:?}", err),
            };
        }
        _ => panic!("Received error response {:?}", response),
    }

    (data.model_name.clone(), start_time.elapsed())
}

#[tokio::main]
async fn main() {
    let mut models: Vec<&dyn Model> = Vec::new();
    models.push(&MaskRCNN {});
    models.push(&Squeezenet {});

    let client = Client::new();
    let mut rng = thread_rng();
    let mut futures: Vec<JoinHandle<(String, Duration)>> = Vec::with_capacity(NUM_REQUESTS);

    for _ in 0..NUM_REQUESTS {
        let model = models.choose(&mut rng).unwrap();
        println!("Sending request for {:?}", &model.name());
        let data = InferenceRequest {
            model_name: model.name(),
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

    analyze_results(elapsed_times);
}
