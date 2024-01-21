/*layer properties:
    neurons


functions:
    activation function
    */

use crate::neural_net::neuron::Neuron;

// potential Enum for tracking which activation function to use
//something like:
/*enum Activation_Function{
    //blah
    //blah
}*/
#[derive(Debug)]
pub struct Layer {
    neurons: Vec<Neuron>,
    // potentially individual bias for each layer
    //option?
    //bias: Option<f64>
    //activation_function: Activation_Function
}

impl Layer {
    pub fn new(num_neurons: usize, num_neurons_in_next_layer: usize) -> Layer {
        let neurons = (0..num_neurons).map(|_| Neuron::new(num_neurons_in_next_layer)).collect();
        
        Layer {
            neurons: neurons
        }
    }

    pub fn get_neurons(&self) -> &Vec<Neuron> {
        &self.neurons
    }
    
    pub fn set_neuron_weights_explicit(&mut self, weights: &Vec<Vec<f64>>) -> Result<(), String>{
        let neuron_len = self.neurons.len();
        let weights_vec_len = weights.len();
        if neuron_len != weights_vec_len {
            return Err(format!("Number of incoming weight vectors does not equal number of neurons. Expected {}, got {}", neuron_len, weights_vec_len));
        }

        let neuron_weights_len = self.neurons[0].get_weights().len();
        let weights_len = weights[0].len();
        if neuron_weights_len != weights_len {
            return Err(format!("Number of neuron weights does not equal incoming weights. Expected {}, got {}", neuron_weights_len, weights_len));
        }

        weights.iter()
            .zip(self.neurons.iter_mut())
            .for_each(|(new_weight_vec, neuron)| neuron.set_weights(new_weight_vec));

        Ok(())
    }

    pub fn sigmoid(sum: f64) -> f64 {
       1.0 / (1.0 + (-sum).exp())
    }

    pub fn sigmoid_derivative(input: f64) -> f64 {
        input * (1.0 - input)
    }

    pub fn feed_forward(&mut self, next_layer: &mut Layer, bias: f64) {

        next_layer.apply_to_neurons(|index, neuron| {

            let sum = self.neurons.iter()
                .fold(0.0, |acc, neuron| acc + (neuron.get_weights()[index] * neuron.get_activation_value())) + bias;

            neuron.set_incoming_value(sum);
            let activated_value = Self::sigmoid(sum);
            neuron.set_activation_value(activated_value);
        });
           }

    pub fn back_propagate(&mut self, previous_layer: &Layer) {
        todo!("write this shit out dawg");
    }

    fn apply_to_neurons<F>(&mut self, mut f: F)
        where F: FnMut(usize, &mut Neuron) {
        for (index, neuron) in &mut self.neurons.iter_mut().enumerate() {
            f(index, neuron);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_initialization_check() {
        let new_layer = Layer::new(5, 5);

        assert_eq!(5, new_layer.get_neurons().len());
        new_layer.get_neurons().iter().for_each(|neuron| assert_eq!(5, neuron.get_weights().len()));
    }

    #[test]
    fn sigmoid_test() {
        let val1 = 0.0;
        let val2 = 0.5;
        let val3 = 1.0;

        assert_eq!(0.5, Layer::sigmoid(val1));
        assert_eq!(0.622, truncate_to_3_decimal_places(Layer::sigmoid(val2)));
        assert_eq!(0.731, truncate_to_3_decimal_places(Layer::sigmoid(val3)));
    }

    #[test]
    fn sigmoid_derivative_test() {
        let val1 = Layer::sigmoid(0.0);
        let val2 = Layer::sigmoid(0.5);
        let val3 = Layer::sigmoid(1.0);

        assert_eq!(0.5, truncate_to_3_decimal_places(val1));
        assert_eq!(0.622, truncate_to_3_decimal_places(val2));
        assert_eq!(0.731, truncate_to_3_decimal_places(val3));

        let sigmoid_derivative1 = Layer::sigmoid_derivative(val1);
        let sigmoid_derivative2 = Layer::sigmoid_derivative(val2);
        let sigmoid_derivative3 = Layer::sigmoid_derivative(val3);

        assert_eq!(0.25, truncate_to_3_decimal_places(sigmoid_derivative1));
        assert_eq!(0.235, truncate_to_3_decimal_places(sigmoid_derivative2));
        assert_eq!(0.197, truncate_to_3_decimal_places(sigmoid_derivative3));
    }

    #[test]
    fn set_weights_explicitly_vec_len_error() {
        let mut layer = Layer::new(1, 4);

        let incorrect_len_vec = vec![
            vec![1.0, 1.0, 1.0, 1.0],
            vec![2.0, 2.0, 2.0, 2.0],
            vec![3.0, 3.0, 3.0, 3.0],
            ];

        let result = layer.set_neuron_weights_explicit(&incorrect_len_vec);

        assert_ne!(Ok(()), result);
        let err_msg = "Number of incoming weight vectors does not equal number of neurons. Expected 1, got 3";
        assert_eq!(err_msg, result.unwrap_err());
    }

    #[test]
    fn set_weights_explicitly_weights_len_error() {
        let mut layer = Layer::new(2, 4);

        let incorrect_weight_len_vec = vec![
            vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0],
            vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            ];

        let result = layer.set_neuron_weights_explicit(&incorrect_weight_len_vec);

        assert_ne!(Ok(()), result);
        let err_msg = "Number of neuron weights does not equal incoming weights. Expected 4, got 6";
        assert_eq!(err_msg, result.unwrap_err());
    }

    #[test]
    fn set_weights_explicitly_correct_lens() {
        let mut layer = Layer::new(3, 5);

        let correct_vec = vec![
            vec![1.0, 1.0, 1.0, 1.0, 1.0],
            vec![2.0, 2.0, 2.0, 2.0, 2.0],
            vec![3.0, 3.0, 3.0, 3.0, 3.0],
            ];

        let result = layer.set_neuron_weights_explicit(&correct_vec);

        assert_eq!(Ok(()), result);
        assert_eq!(correct_vec[0], layer.get_neurons()[0].get_weights());
        assert_eq!(correct_vec[1], layer.get_neurons()[1].get_weights());
        assert_eq!(correct_vec[2], layer.get_neurons()[2].get_weights());
    } 

    #[test]
    fn feed_forward_input_to_hidden_layer_check() {
        let mut input_layer = Layer::new(2, 4);
        let mut hidden_layer = Layer::new(4, 1);
        input_layer.neurons[0].set_incoming_value(0.6);
        input_layer.neurons[0].set_activation_value(0.6);
        input_layer.neurons[1].set_incoming_value(0.4);
        input_layer.neurons[1].set_activation_value(0.4);


        let input_layer_weights = vec![
            vec![0.15, 0.2, 0.25, 0.3],
            vec![0.35, 0.4, 0.45, 0.5],
        ];

        let result = input_layer.set_neuron_weights_explicit(&input_layer_weights);
        assert_eq!(Ok(()), result);
        assert_eq!(input_layer_weights[0], input_layer.get_neurons()[0].get_weights());
        assert_eq!(input_layer_weights[1], input_layer.get_neurons()[1].get_weights());

        input_layer.feed_forward(&mut hidden_layer, 0.5);

        // i could put these next 8 assertions in a loop, but i like the clarity here
        assert_eq!(0.730, truncate_to_3_decimal_places(hidden_layer.get_neurons()[0].get_incoming_value()));
        assert_eq!(0.780, truncate_to_3_decimal_places(hidden_layer.get_neurons()[1].get_incoming_value()));
        assert_eq!(0.830, truncate_to_3_decimal_places(hidden_layer.get_neurons()[2].get_incoming_value()));
        assert_eq!(0.880, truncate_to_3_decimal_places(hidden_layer.get_neurons()[3].get_incoming_value()));

        assert_eq!(0.675, truncate_to_3_decimal_places(hidden_layer.get_neurons()[0].get_activation_value()));
        assert_eq!(0.686, truncate_to_3_decimal_places(hidden_layer.get_neurons()[1].get_activation_value()));
        assert_eq!(0.696, truncate_to_3_decimal_places(hidden_layer.get_neurons()[2].get_activation_value()));
        assert_eq!(0.707, truncate_to_3_decimal_places(hidden_layer.get_neurons()[3].get_activation_value()));
    }  


    fn truncate_to_3_decimal_places(val: f64) -> f64 {
        let scale = 1000.0;
        (val * scale).round() / scale
    }
}


