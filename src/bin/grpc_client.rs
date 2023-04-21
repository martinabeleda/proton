use rand::prelude::*;
use std::error::Error;
use tonic::Request;

use proton::predictor::predictor_client::PredictorClient;
use proton::predictor::InferenceRequest;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let mut client = PredictorClient::connect("http://0.0.0.0:50051").await?;

    let input_shape = vec![1, 3, 224, 224];
    let n = input_shape.iter().product();

    let mut rng = rand::thread_rng();
    let random_vector: Vec<f32> = (0..n).map(|_| rng.gen()).collect();

    let request = Request::new(InferenceRequest {
        model_name: "squeezenet".into(),
        data: random_vector,
        shape: input_shape,
    });

    match client.predict(request).await {
        Ok(res) => {
            let response = res.into_inner();
            println!(
                "Success for model {}, prediction_id={}",
                response.model_name, response.prediction_id
            );
        }
        Err(err) => println!("Error {:?}", err),
    }

    Ok(())
}
