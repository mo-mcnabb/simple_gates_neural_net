use crate::neural_net::neuron::Neuron;

// potential Enum for tracking which activation function to use
//something like:
/*enum Activation_Function{
    //blah
    //blah
}*/
#[derive(Debug, Clone)]
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

    pub fn get_neurons_mut(&mut self) -> &mut Vec<Neuron> {
        &mut self.neurons
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

    pub fn set_neuron_values_explicit(&mut self, values: Vec<f64>) -> Result<(), String> {
        if self.get_neurons().len() != values.len() {
            return Err(format!("Number of neurons does not equal the number of values given to the function. Expected {}, got {}", self.get_neurons().len(), values.len()));
        }

        self.neurons.iter_mut().zip(values).for_each(|(neuron, value)| {
            neuron.set_incoming_value(value);
            neuron.set_activation_value(value);
        });

        Ok(())
    }

    pub fn sigmoid(sum: f64) -> f64 {
       1.0 / (1.0 + (-sum).exp())
    }

    pub fn sigmoid_derivative(input: f64) -> f64 {
        Layer::sigmoid(input) * (1.0 - Layer::sigmoid(input))
    }

    pub fn sigmoid_derivative_already_activated(activated_value: f64) -> f64 {
        activated_value * (1.0 - activated_value)
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

    pub fn set_default_neuron_values(&mut self, value: f64) {
        self.neurons.iter_mut().for_each(|neuron| {
            
            neuron.set_activation_value(value);
            neuron.set_incoming_value(value);
        });
    }

    pub fn set_neuron_error_gradients(&mut self, error: f64) {
        
        self.neurons.iter_mut().for_each(|neuron| {
            let sigmoid_d = Layer::sigmoid_derivative(neuron.get_activation_value());
            
            neuron.set_error_gradient(sigmoid_d * error);
        });
    }

    pub fn back_propagate(&mut self, expected_values: &Vec<f64>, learning_rate: f64, next_layer: Option<&Layer>) {
        if let Some(next_layer) = next_layer {
            // Check if next_layer is the output layer
            if next_layer.get_neurons()[0].get_weights().len() == 0 {
                // output layer -> hidden layer case
                let weights: Vec<Vec<f64>> = self.get_neurons().iter()
                    .map(|neuron| neuron.get_weights().clone())
                    .collect();

                for (i, neuron) in self.get_neurons_mut().iter_mut().enumerate() {
                    let error: f64 = next_layer.neurons.iter().enumerate().fold(0.0, |acc, (j, next_neuron)| {
                        acc + (weights[i][j] * next_neuron.get_error_gradient())
                    });

                    
                    let sigmoid_deriv = Layer::sigmoid_derivative_already_activated(neuron.get_activation_value());
                    neuron.set_error_gradient(sigmoid_deriv * error);
                }
            } 
            else {
                // Hidden layer case
                for (i, neuron) in self.neurons.iter_mut().enumerate() {
                    let error: f64 = next_layer.neurons.iter().fold(0.0, |acc, next_neuron| {
                        acc + (next_neuron.get_weights()[i] * next_neuron.get_error_gradient())
                    });

                    let sigmoid_deriv = Layer::sigmoid_derivative(neuron.get_activation_value());
                    neuron.set_error_gradient(sigmoid_deriv * error);
                }
            }
        } 
        else {
            // Output layer case
            for (neuron, &expected) in self.get_neurons_mut().iter_mut().zip(expected_values) {
                let error = expected - neuron.get_activation_value();
                let sigmoid_deriv = Layer::sigmoid_derivative_already_activated(neuron.get_activation_value());

                // the cost function is (OUTPUTexpected - OUTPUTpredicted)^2
                // we calculate the inside of the parentheses above with the error variable
                // the derivative of (OUTPUTexpected - OUTPUTpredicted)^2 = 2(OUTPUTexpected - OUTPUTpredicted)
                // so the derivative of the cost function is 2.0 * error
                // and the error gradient is derivative of the cost function * the sigmoid derivative :D
                neuron.set_error_gradient(2.0 *  error * sigmoid_deriv);
            }
        }

        // Update weights
        self.update_weights_back_prop(learning_rate);
    }


    //error_gradient = 0.2
    //activation_value = 5.0
    // weights: [2.0, 4.0]
    fn update_weights_back_prop(&mut self, learning_rate: f64) {
         self.get_neurons_mut()
            .iter_mut()
            .for_each(|neuron| {
                let old_weights = neuron.get_weights();
                let mut new_weights: Vec<f64> = Vec::new();

                old_weights.iter().for_each(|old_weight| {
                    let val = old_weight + (learning_rate * neuron.get_error_gradient() * neuron.get_activation_value());
                    new_weights.push(val);
                });
                neuron.set_weights(&new_weights);
            });
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

        let sigmoid_derivative1 = Layer::sigmoid_derivative(0.0);
        let sigmoid_derivative2 = Layer::sigmoid_derivative(0.5);
        let sigmoid_derivative3 = Layer::sigmoid_derivative(1.0);

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

    #[test]
    fn feed_forward_last_pass_hidden_layer_to_output() {
        let mut hidden_layer = Layer::new(4, 1);
        hidden_layer.neurons[0].set_incoming_value(0.730);
        hidden_layer.neurons[0].set_activation_value(0.675);
        hidden_layer.neurons[1].set_incoming_value(0.780);
        hidden_layer.neurons[1].set_activation_value(0.686);
        hidden_layer.neurons[2].set_incoming_value(0.830);
        hidden_layer.neurons[2].set_activation_value(0.696); 
        hidden_layer.neurons[3].set_incoming_value(0.880);
        hidden_layer.neurons[3].set_activation_value(0.707);

        let hidden_layer_weights = vec![
            vec![0.40],
            vec![0.45],
            vec![0.50],
            vec![0.55]
        ];

        let result = hidden_layer.set_neuron_weights_explicit(&hidden_layer_weights);

        assert_eq!(Ok(()), result);

        let mut output_layer = Layer::new(1, 0);

        hidden_layer.feed_forward(&mut output_layer, 0.5);

        assert_eq!(1.816, truncate_to_3_decimal_places(output_layer.get_neurons()[0].get_incoming_value()));
        assert_eq!(0.860, truncate_to_3_decimal_places(output_layer.get_neurons()[0].get_activation_value()));
    }

    #[test]
    fn set_default_neuron_values_test() {
        let mut layer = Layer::new(3, 0);

        layer.set_default_neuron_values(0.5);

        layer.get_neurons().iter().for_each(|neuron| {
            
            assert_eq!(0.5, neuron.get_activation_value());
            assert_eq!(0.5, neuron.get_incoming_value());
        });

        layer.set_default_neuron_values(9.0);

        layer.get_neurons().iter().for_each(|neuron| {
            
            assert_eq!(9.0, neuron.get_activation_value());
            assert_eq!(9.0, neuron.get_incoming_value());
        });
    }

    #[test]
    fn set_neuron_error_gradients_check() {
        let mut layer = Layer::new(3, 0); 
        layer.neurons[0].set_activation_value(0.0);
        layer.neurons[1].set_activation_value(0.0);
        layer.neurons[2].set_activation_value(0.0);

        // sigmoid(input) * (1 - sigmoid(input))
        let error = -4.0;
        layer.set_neuron_error_gradients(error);

        assert_eq!(-1.0, layer.get_neurons()[0].get_error_gradient());
        assert_eq!(-1.0, layer.get_neurons()[1].get_error_gradient());
        assert_eq!(-1.0, layer.get_neurons()[2].get_error_gradient());
    }

    #[test]
    fn set_neuron_values_explicit_incorrect_len() {
        let mut layer = Layer::new(3, 0);
        let incorrect_len_values = vec![
            3.0,
            4.0,
            5.0,
            6.0,
            7.0
        ];

        let result = layer.set_neuron_values_explicit(incorrect_len_values);

        assert_ne!(Ok(()), result);
        assert_eq!("Number of neurons does not equal the number of values given to the function. Expected 3, got 5", result.unwrap_err());
    }

    #[test]
    fn set_neuron_values_explicit_check() {
        let mut layer = Layer::new(3, 0);
        let correct_len_values = vec![
            3.0,
            4.0,
            5.0,
        ];

        let result = layer.set_neuron_values_explicit(correct_len_values);

        assert_eq!(Ok(()), result);

        assert_eq!(3.0, layer.get_neurons()[0].get_activation_value());
        assert_eq!(3.0, layer.get_neurons()[0].get_incoming_value());
        assert_eq!(4.0, layer.get_neurons()[1].get_activation_value());
        assert_eq!(4.0, layer.get_neurons()[1].get_incoming_value());
        assert_eq!(5.0, layer.get_neurons()[2].get_activation_value());
        assert_eq!(5.0, layer.get_neurons()[2].get_incoming_value());
    }

    #[test]
    fn output_layer_back_propagation_test(){
        let mut output_layer = Layer::new(1,0);
        let result = output_layer.set_neuron_values_explicit(vec![0.75]);
        assert_eq!(Ok(()), result);

        // corresponds to number of output neurons
        let expected_values = vec![0.7];

        // None because there is no next layer - the output is the last
        output_layer.back_propagate(&expected_values, 0.01, None);

        assert_eq!(-0.019, truncate_to_3_decimal_places(output_layer.get_neurons()[0].get_error_gradient()));
    }

    #[test]
    fn hidden_layer_back_propagation_test() {
        let mut hidden_layer = Layer::new(4, 1);
        let mut output_layer = Layer::new(1, 0);
        let result = output_layer.set_neuron_values_explicit(vec![0.75]);
        assert_eq!(Ok(()), result);

        let hidden_layer_values = vec![0.6, 0.7, 0.8, 0.9];
        let result = hidden_layer.set_neuron_values_explicit(hidden_layer_values);
        assert_eq!(Ok(()), result);
    
        let hidden_layer_initial_weights = vec![
            vec![0.9],
            vec![0.1],
            vec![0.8],
            vec![0.2],
        ];
        let result = hidden_layer.set_neuron_weights_explicit(&hidden_layer_initial_weights);
        assert_eq!(Ok(()), result);
        
        let expected_values = vec![0.7];
        output_layer.back_propagate(&expected_values, 0.01, None);
        assert_eq!(-0.019, truncate_to_3_decimal_places(output_layer.get_neurons()[0].get_error_gradient()));

        hidden_layer.back_propagate(&expected_values, 0.01, Some(&output_layer));
 
        assert_eq!(-0.0041, truncate_to_4_decimal_places(hidden_layer.get_neurons()[0].get_error_gradient()));
        assert_eq!(-0.0004, truncate_to_4_decimal_places(hidden_layer.get_neurons()[1].get_error_gradient()));
        assert_eq!(-0.0024, truncate_to_4_decimal_places(hidden_layer.get_neurons()[2].get_error_gradient()));
        assert_eq!(-0.0003, truncate_to_4_decimal_places(hidden_layer.get_neurons()[3].get_error_gradient()));

        assert_eq!(0.900, truncate_to_3_decimal_places(hidden_layer.get_neurons()[0].get_weights()[0]));
        assert_eq!(0.100, truncate_to_3_decimal_places(hidden_layer.get_neurons()[1].get_weights()[0]));
        assert_eq!(0.800, truncate_to_3_decimal_places(hidden_layer.get_neurons()[2].get_weights()[0]));
        assert_eq!(0.200, truncate_to_3_decimal_places(hidden_layer.get_neurons()[3].get_weights()[0]));

        // corresponds to number of output neurons
     }

    #[test]
    fn update_weights_back_prop_check() {
        let mut layer = Layer::new(1, 2);
        layer.get_neurons_mut().iter_mut().for_each(|neuron| {
            
            neuron.set_error_gradient(0.2);
            neuron.set_activation_value(5.0);

            let weights = vec![2.0, 4.0];
            neuron.set_weights(&weights);
        });

        layer.update_weights_back_prop(0.1);

        assert_eq!(2.1, layer.get_neurons()[0].get_weights()[0]);
        assert_eq!(4.1, layer.get_neurons()[0].get_weights()[1]);
    }

    fn truncate_to_3_decimal_places(val: f64) -> f64 {
        let scale = 1000.0;
        (val * scale).round() / scale
    }

    fn truncate_to_4_decimal_places(val: f64) -> f64 {
        let scale = 10_000.0;
        (val * scale).round() / scale
    }
}