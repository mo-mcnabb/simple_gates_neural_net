use crate::neural_net::layer::Layer;
pub struct NeuralNet {
    inputs: Vec<(i32, i32, i32)>,
    set_name: String,
    input_layer: Layer,
    hidden_layers: Vec<Layer>,
    output_layer: Layer,
    bias: f64, 
    learning_rate: f64,
}

impl NeuralNet {
    pub fn new(inputs: Vec<(i32, i32, i32)>, set_name: String, num_hidden_layers: usize, num_hidden_layer_neurons: usize, num_outputs: usize, bias: f64, learning_rate: f64) -> NeuralNet {
        
        let mut hidden_layers: Vec<Layer> = (0..num_hidden_layers).map(|_| Layer::new(num_hidden_layer_neurons)).collect();
        let output_layer = Layer::new(num_outputs);
        let input_layer = Layer::new(2);

        if hidden_layers.len() > 1 {
            for index in 0..(hidden_layers.len() - 1) {
                let next_layer = hidden_layers.get(index + 1).unwrap().clone();
                if let Some(layer) = hidden_layers.get_mut(index) {
                    layer.init_weights(&next_layer);
                }
            }
        }

        // sets the weights for the neurons with respect to the outputs neurons,
        // as this is indexing the last layer of the neural network
        let last_layer_index = hidden_layers.len() - 1;
        hidden_layers.get_mut(last_layer_index).unwrap().init_weights(&output_layer);

        NeuralNet {
            inputs: inputs,
            set_name: set_name,
            input_layer: input_layer,
            hidden_layers: hidden_layers,
            output_layer: output_layer,
            bias: bias,
            learning_rate: learning_rate,
        }
    }
    
    pub fn train(&mut self, iterations: usize) {
        let inputs = self.inputs.clone();
        let mut hidden_layers: Vec<Layer> = self.hidden_layers.clone();

        for _ in 0..iterations {

            for (x, y, answer) in &inputs {

                self.input_layer.set_neuron_value(0, *x as f64);
                self.input_layer.set_neuron_value(1, *y as f64);

                Self::feed_forward(hidden_layers.get_mut(0).unwrap(), &mut self.input_layer, self.bias);

                for index in 1..(self.hidden_layers.len()) {

                    let (left, right) = hidden_layers.split_at_mut(index);
                    let previous_layer = left.last_mut().unwrap();
                    let current_layer = right.first_mut().unwrap();
                    Self::feed_forward(current_layer, previous_layer, self.bias);
                }

                Self::feed_forward(&mut self.output_layer, &mut self.hidden_layers.get_mut(hidden_layers.len() - 1).unwrap(), self.bias);

                // TODO: add logic for handling more than one output neuron
                let output = self.output_layer.neurons.get(0).unwrap().value;
                let sigmoid_derivative = Self::sigmoid_derivative(output);
                let error = *answer as f64 - output;

                Self::back_propagate_bias(self, &error, &sigmoid_derivative);
                self.hidden_layers.iter_mut().for_each(|layer| Self::back_propagate(layer, &error, &sigmoid_derivative, &self.learning_rate));
                Self::back_propagate(&mut self.input_layer, &error, &sigmoid_derivative, &self.learning_rate);

            }
        }
        self.hidden_layers = hidden_layers;
    }

    fn feed_forward(current_layer: &mut Layer, previous_layer: &mut Layer, bias: f64) {

        for (index, neuron) in current_layer.neurons.iter_mut().enumerate() {
            let mut weighted_sum = 0.0;
    
            for prev_neuron in previous_layer.neurons.iter() {
                if let Some(weight) = prev_neuron.weights.get(index) {
                    let val = prev_neuron.value;
                    weighted_sum += val * weight;
                }
            }
    
            neuron.value = Self::sigmoid(weighted_sum + bias);
        }
    }

    fn back_propagate(layer: &mut Layer, error: &f64, sigmoid_derivative: &f64, learning_rate: &f64) {
        for neuron in layer.neurons.iter_mut() {
            let weight_gradient = neuron.value * error * sigmoid_derivative;
            neuron.weights = neuron.weights.iter_mut().map(|weight| *weight + (learning_rate * weight_gradient)).collect();
        } 
    }

    fn back_propagate_bias(&mut self, error: &f64, sigmoid_derivative: &f64) {
        let bias_gradient = error * sigmoid_derivative;
        self.bias = self.bias + (self.learning_rate * bias_gradient);
    }

    fn sigmoid(sum: f64) -> f64 {
        1.0 / (1.0 + (-sum).exp())
    }

    fn sigmoid_derivative(output: f64) -> f64 {
        output * (1 as f64 - output)
    }
}