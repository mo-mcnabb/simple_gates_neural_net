use rand::prelude::*;
pub struct Neuron {
    value: f64,
    weights: Vec<f64>
}

impl Neuron {
    pub fn new(start_value: usize) -> Neuron {
       
        Neuron {
            value: start_value,
        }
    }

    pub fn set_value(&mut self, value: f64) {
        self.value = value;
    }

    pub fn init_weights(&mut self, num_neurons_in_next_layer: usize) {
        let mut rng = rand::thread_rng();
        let weights: Vec<f64> = (0..num_neroun_in_next_layer).map(|_| rng.gen()).collect(); 
    }
}

