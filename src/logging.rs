use std::str::FromStr;
use tracing::Level;

pub fn setup(level: &String) {
    let log_level = match Level::from_str(level) {
        Ok(log_level) => log_level,
        Err(_) => panic!("Unsupported log level {:?}", level),
    };
    tracing_subscriber::fmt()
        .with_max_level(log_level)
        .with_file(true)
        .with_line_number(true)
        .init();
}
