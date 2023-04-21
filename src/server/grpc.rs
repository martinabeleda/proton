use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::mpsc::Sender;
use tonic::transport::{Error, Server};
use tonic::{Request, Response, Status};

use crate::predictor::predictor_server::{Predictor, PredictorServer};
use crate::predictor::{InferenceRequest, InferenceResponse};
use crate::worker::Message;

#[derive(Debug, Default)]
pub struct PredictService {
    queues_tx: Arc<HashMap<String, Sender<Message>>>,
}

impl PredictService {
    fn new(queues_tx: Arc<HashMap<String, Sender<Message>>>) -> Self {
        Self { queues_tx }
    }
}

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

pub async fn build(
    port: u16,
    queues_tx: Arc<HashMap<String, Sender<Message>>>,
) -> Result<(), Error> {
    let grpc_addr = SocketAddr::from(([0, 0, 0, 0], port));

    let predict_service = PredictService::new(queues_tx);

    Server::builder()
        .add_service(PredictorServer::new(predict_service))
        .serve(grpc_addr)
        .await
}
