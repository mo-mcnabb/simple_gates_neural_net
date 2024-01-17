pub struct Layer {
    neurons: Vec<Neurons>
}

impl Layer {
    pub fn new(num_neurons: usize) -> Layer {
        let neurons = (0..num_neurons).map(|_| Neurons::new(0));
        Layer {
            nuerons: neurons,
        }
    }

    fn feed_forward(&mut self, previous_layer: &mut Layer, bias: f64) {
        for (index, neuron) in current_layer.neurons.iter_mut().enumerate() {
            let mut weighted_sum = 0.0;
    
            for prev_neuron in previous_layer.neurons.iter() {
                if let Some(weight) = prev_neuron.weights.get(index) {
                    let val = prev_neuron.value;
                    weighted_sum += val * weight;
                }
            }
    
            neuron.value = weighted_sum + bias;
        }
    }
    
    fn back_propagate(layer: Layer, error: f64, sigmoid_derivative: f64, learning_rate: f64) {
        for(index, neuron) in layer {
            let weight_gradient = neuron.value * error * sigmoid_derivative;
            neuron.weights.iter_mut().for_each(|x| x + (learning_rate * weight_gradient));
        } 
    
    }

    fn sigmoid(sum: f64) -> f64 {
        1.0 / (1.0 + (-sum).exp())
    }

    fn sigmoid_derivative(output: f64) -> f64 {
        output * (1 as f64 - output)
    }

    fn set_weights(&mut self, next_layer: Layer) {
        //for neuron in self.neurons {
            //set neuron vec to num neurons in next layer
            //set neuron weights to random float
        //}
    }


}