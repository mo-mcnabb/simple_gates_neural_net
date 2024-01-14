use rand::prelude::*;
pub struct Neuron {
    weights: Vec<f64>
}

impl Neuron {
    pub fn new(num_neurons_in_next_layer: usize) -> Neuron {
        let mut rng = rand::thread_rng();
        let weights: Vec<f64> = (0..num_neroun_in_next_layer).map(|_| rng.gen()).collect(); 
        
        Neuron {
            weights: weights,
        }
    }
}

