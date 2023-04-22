use ndarray::{Array, Dimension, IxDyn};
use std::collections::HashMap;
use std::time::Duration;

pub trait Model {
    fn name(&self) -> String;

    fn input_shape(&self) -> Vec<usize>;

    fn dummy_data(&self) -> Array<f32, IxDyn> {
        let shape = IxDyn(&self.input_shape()[..]);
        Array::linspace(0.0_f32, 1.0, shape.size())
            .into_shape(shape)
            .unwrap()
    }
}

pub struct Squeezenet;

impl Model for Squeezenet {
    fn name(&self) -> String {
        "squeezenet".to_string()
    }

    fn input_shape(&self) -> Vec<usize> {
        vec![1, 3, 224, 224]
    }
}

pub struct MaskRCNN;

impl Model for MaskRCNN {
    fn name(&self) -> String {
        "maskrcnn".to_string()
    }

    fn input_shape(&self) -> Vec<usize> {
        vec![3, 224, 224]
    }
}

pub fn analyze_results(elapsed_times: HashMap<String, Vec<Duration>>) {
    for (model_name, times) in &elapsed_times {
        let total_time: Duration = times.iter().cloned().sum();
        let average_time = total_time / times.len() as u32;
        let mut sorted_times = times.clone();
        sorted_times.sort_unstable();

        let p50 = sorted_times[((0.50 * (times.len() as f64)).round() as usize) - 1];
        let p95 = sorted_times[((0.95 * (times.len() as f64)).round() as usize) - 1];
        let p99 = sorted_times[((0.99 * (times.len() as f64)).round() as usize) - 1];

        println!("\nModel: {:?}", model_name);
        println!("Total time: {:?}", total_time);
        println!("Average time: {:?}", average_time);
        println!("Median time: {:?}", p50);
        println!("p95 time: {:?}", p95);
        println!("p99 time: {:?}", p99);
    }
}
