use rand::prelude::*;
use std::collections::HashMap;
use std::time::Duration;
use tokio::task::JoinHandle;
use tonic::codec::CompressionEncoding;
use tonic::transport::Channel;
use tonic::Request;

use proton::predictor::predictor_client::PredictorClient;
use proton::predictor::InferenceRequest;
use proton::utils::{analyze_results, MaskRCNN, Model, Squeezenet};

const NUM_REQUESTS: usize = 20;

async fn send_request(
    client: &mut PredictorClient<Channel>,
    request: Request<InferenceRequest>,
) -> (String, Duration) {
    let start_time = tokio::time::Instant::now();

    let response = client.predict(request).await.unwrap().into_inner();
    println!(
        "Success for model {}, prediction_id={}",
        response.model_name, response.prediction_id
    );

    (response.model_name, start_time.elapsed())
}

#[tokio::main]
async fn main() {
    let mut models: Vec<&dyn Model> = Vec::new();
    models.push(&MaskRCNN {});
    models.push(&Squeezenet {});

    let channel = Channel::builder("http://0.0.0.0:50051".parse().unwrap())
        .connect()
        .await
        .unwrap();

    let client = PredictorClient::new(channel)
        .send_compressed(CompressionEncoding::Zstd)
        .accept_compressed(CompressionEncoding::Zstd);

    let mut rng = thread_rng();
    let mut futures: Vec<JoinHandle<(String, Duration)>> = Vec::with_capacity(NUM_REQUESTS);

    for _ in 0..NUM_REQUESTS {
        let model = models.choose(&mut rng).unwrap();
        let n = model.input_shape().iter().product();
        let random_vector: Vec<f32> = (0..n).map(|_| rng.gen()).collect();

        println!("Sending request for {:?}", &model.name());
        let request = Request::new(InferenceRequest {
            model_name: model.name(),
            data: random_vector,
            shape: model.input_shape().iter().map(|x| *x as i32).collect(),
        });

        let mut client = client.clone();
        let task = tokio::spawn(async move { send_request(&mut client, request).await });
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
