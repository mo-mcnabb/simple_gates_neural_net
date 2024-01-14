use rand::prelude::*;
pub struct Neuron {
    weights: Vec<f64>
}

impl Neuron {
    pub fn new(num_weights: usize) -> Neuron {
        let mut rng = rand::thread_rng();
        let weights: Vec<f64> = (0..100).map(|_| rng.gen()).collect(); 
        
        Neuron {
            weights: weights,
        }
        
    }
}

