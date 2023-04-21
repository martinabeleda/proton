use std::error::Error;
use tonic::Request;

use proton::predictor::predictor_client::PredictorClient;
use proton::predictor::InferenceRequest;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let mut client = PredictorClient::connect("http://0.0.0.0:50051").await?;

    let request = Request::new(InferenceRequest {
        model_name: "squeezenet".into(),
        data: vec![1.0, 2.0, 3.0],
        shape: vec![3],
    });

    let response = client.predict(request).await?;

    println!("RESPONSE={:?}", response);

    Ok(())
}
