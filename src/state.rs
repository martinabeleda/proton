use std::collections::HashMap;

use crate::config::Config;

pub struct SharedState {
    pub config: Config,
    pub ready: HashMap<String, bool>,
}

impl SharedState {
    pub fn new(config: Config) -> Self {
        let ready: HashMap<String, bool> = config
            .models
            .iter()
            .map(|model_config| {
                // Initialize models as not ready. InferenceWorker
                // is responsible for updating these flags once models come online
                (model_config.name.clone(), false)
            })
            .collect();

        SharedState { config, ready }
    }
}
