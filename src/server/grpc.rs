use ndarray::{Array, IxDyn};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::mpsc::Sender;
use tokio::sync::oneshot;
use tonic::transport::{Error, Server};
use tonic::{Request, Response, Status};
use uuid::Uuid;

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
        let request = request.into_inner();

        let prediction_id = Uuid::new_v4();

        // Create a channel to receive the inference result
        let (tx, rx) = oneshot::channel();

        let input_shape: Vec<usize> = request.shape.iter().map(|&value| value as usize).collect();

        let input_shape = IxDyn(&input_shape[..]);

        let input_data = Array::from_shape_vec(input_shape, request.data).unwrap();

        let message = Message {
            prediction_id,
            input_data,
            model_name: request.model_name.clone(),
            response_tx: tx,
        };

        self.queues_tx
            .get(&request.model_name)
            .unwrap()
            .send(message)
            .await
            .unwrap();

        let response = rx.await.unwrap();

        tracing::info!(
            "gRPC Handler received prediction_id={:?} for model={}",
            prediction_id,
            &request.model_name
        );

        let shape = response
            .shape()
            .to_vec()
            .iter()
            .map(|&value| value as i32)
            .collect();

        let response = InferenceResponse {
            model_name: request.model_name,
            prediction_id: prediction_id.to_string(),
            data: response.into_raw_vec(),
            shape,
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
