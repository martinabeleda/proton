use serde::Deserialize;
use std::error::Error;
use std::fs::read_to_string;

#[derive(Clone, Debug, Deserialize)]
pub struct ModelConfig {
    pub name: String,
    pub path: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ServerConfig {
    pub num_threads: i16,
    pub buffer_size: usize,
    pub port: u16,
    pub grpc_port: u16,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Config {
    pub log_level: String,
    pub models: Vec<ModelConfig>,
    pub server: ServerConfig,
}

impl Config {
    pub fn load(path: &str) -> Result<Config, Box<dyn Error>> {
        let config_data = read_to_string(path)?;
        let config: Config = serde_yaml::from_str(&config_data)?;

        Ok(config)
    }
}
