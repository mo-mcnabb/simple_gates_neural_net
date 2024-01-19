use crate::neural_net::neuron::Neuron;

#[derive(Clone)]
pub struct Layer {
    pub neurons: Vec<Neuron>
}

impl Layer {
    pub fn new(num_neurons: usize) -> Layer {
        let mut neurons = (0..num_neurons).map(|_| Neuron::new(0.0)).collect();
        
        Layer {
            neurons: neurons,
        }
    }

    pub fn init_weights(&mut self, next_layer: &Layer) {
        self.neurons.iter_mut().for_each(|neuron| neuron.init_weights(next_layer.neurons.len()));
    }

    pub fn set_neuron_value(&mut self, index: usize, val: f64) {
        if let Some(neuron) = self.neurons.get_mut(index) {
            neuron.set_value(val);
        }
    }
}