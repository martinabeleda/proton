#[macro_use]
extern crate lazy_static;

pub mod config;
pub mod logging;
pub mod model;
pub mod routes;
pub mod server;
pub mod state;
pub mod worker;

pub mod predictor {
    tonic::include_proto!("predict");
}
