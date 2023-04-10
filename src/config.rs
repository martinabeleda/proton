use serde::Deserialize;
use std::error::Error;
use tokio::fs;

#[derive(Clone, Debug, Deserialize)]
pub struct ModelConfig {
    pub name: String,
    pub path: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ServerConfig {
    pub buffer_size: usize,
    pub port: u16,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Config {
    pub models: Vec<ModelConfig>,
    pub server: ServerConfig,
}

impl Config {
    pub async fn load(path: &str) -> Result<Config, Box<dyn Error>> {
        let config_data = fs::read_to_string(path).await?;
        let config: Config = serde_yaml::from_str(&config_data)?;
        Ok(config)
    }
}
