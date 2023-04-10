use serde::Deserialize;

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
