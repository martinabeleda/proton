#[macro_use]
extern crate lazy_static;

pub mod config;
pub mod logging;
pub mod model;
pub mod routes;
pub mod state;
pub mod worker;

pub mod grpc {
    tonic::include_proto!("predict");
}
