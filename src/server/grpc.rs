use std::net::SocketAddr;
use tonic::transport::{Error, Server};
use tonic::{Request, Response, Status};

use crate::predictor::predictor_server::{Predictor, PredictorServer};
use crate::predictor::{InferenceRequest, InferenceResponse};

#[derive(Debug, Default)]
pub struct PredictService;

#[tonic::async_trait]
impl Predictor for PredictService {
    async fn predict(
        &self,
        request: Request<InferenceRequest>,
    ) -> Result<Response<InferenceResponse>, Status> {
        let _input = request.into_inner();

        let response = InferenceResponse {
            model_name: String::from("my_model"),
            prediction_id: String::from("some_id"),
            data: vec![0.0, 1.0],
            shape: vec![2],
        };

        Ok(Response::new(response))
    }
}

pub async fn build(port: u16) -> Result<(), Error> {
    let grpc_addr = SocketAddr::from(([0, 0, 0, 0], port));

    Server::builder()
        .add_service(PredictorServer::new(PredictService {}))
        .serve(grpc_addr)
        .await
}
