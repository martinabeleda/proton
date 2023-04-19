use std::collections::HashMap;
use std::sync::atomic::AtomicBool;

use crate::config::Config;

#[derive(Debug)]
pub struct SharedState {
    pub config: Config,
    pub ready: HashMap<String, AtomicBool>,
}

impl SharedState {
    pub fn new(config: Config) -> Self {
        let ready: HashMap<String, AtomicBool> = config
            .models
            .iter()
            .map(|model_config| {
                // Initialize models as not ready. InferenceWorker
                // is responsible for updating these flags once models come online
                (model_config.name.clone(), AtomicBool::new(false))
            })
            .collect();

        SharedState { config, ready }
    }
}
