use ndarray::{Array, IxDyn};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::mpsc::Sender;
use tokio::sync::oneshot;
use tonic::codec::CompressionEncoding;
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
        let prediction_id = Uuid::new_v4();
        let request = request.into_inner();
        let model_name = request.model_name.clone();
        tracing::info!(
            "gRPC handler created prediction_id={:?} for model={}",
            prediction_id,
            &model_name
        );

        // Create a channel to receive the inference result
        let (response_tx, response_rx) = oneshot::channel();

        let input_shape: Vec<usize> = request
            .shape
            .iter()
            .map(|&value| value as usize)
            .collect::<Vec<usize>>();
        let input_data = Array::from_shape_vec(IxDyn(&input_shape[..]), request.data).unwrap();

        let message = Message {
            prediction_id,
            input_data,
            response_tx,
            model_name: model_name.clone(),
        };

        // Send the prediction to the queue for this model
        self.queues_tx
            .get(&model_name)
            .unwrap()
            .send(message)
            .await
            .unwrap();

        // Wait for the prediction
        let response = response_rx.await.unwrap();
        tracing::info!(
            "gRPC handler received prediction_id={:?} for model={}",
            prediction_id,
            &model_name
        );

        let shape = response
            .shape()
            .to_vec()
            .iter()
            .map(|&value| value as i32)
            .collect();

        Ok(Response::new(InferenceResponse {
            model_name: request.model_name,
            prediction_id: prediction_id.to_string(),
            data: response.into_raw_vec(),
            shape,
        }))
    }
}

pub async fn build(
    port: u16,
    queues_tx: Arc<HashMap<String, Sender<Message>>>,
) -> Result<(), Error> {
    let grpc_addr = SocketAddr::from(([0, 0, 0, 0], port));
    tracing::info!("Starting gRPC server, binding to port {:?}", port);

    let predict_service = PredictService::new(queues_tx);

    Server::builder()
        .add_service(
            PredictorServer::new(predict_service)
                .accept_compressed(CompressionEncoding::Zstd)
                .send_compressed(CompressionEncoding::Zstd),
        )
        .serve(grpc_addr)
        .await
}
