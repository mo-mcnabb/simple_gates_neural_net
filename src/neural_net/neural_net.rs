/*neural net properties:
    layers: vec of layers
    bias
    learning rate
*/

use crate::neural_net::layer::Layer;
use rand::prelude::Rng;
struct NeuralNet {
    layers: Vec<Layer>,
    bias: f64,
    learning_rate: f64,
}

impl NeuralNet {
    pub fn new(num_layers: usize, num_inputs: usize, num_hidden_layer_neurons: usize, num_outputs: usize,learning_rate: f64) -> NeuralNet {
        let mut layers: Vec<Layer> = Vec::new();
        
        // input layer
        layers.push(Layer::new(num_inputs, num_hidden_layer_neurons));
        
        // ignoring input layer and output layer
        let num_hidden_layers = num_layers - 2;
        for i in 0..num_hidden_layers {
           if i == num_hidden_layers - 1 {
            // hidden layer  -> output layer
                layers.push(Layer::new(num_hidden_layer_neurons, num_outputs));
                continue;
           } 

           layers.push(Layer::new(num_hidden_layer_neurons, num_hidden_layer_neurons));
        }

        layers.push(Layer::new(num_outputs, 0));

        let mut rng = rand::thread_rng();
        let bias: f64 = rng.gen();

        NeuralNet {
            layers: layers,
            bias: bias,
            learning_rate: learning_rate,
        }
    }

    // mean square error cost function, idk
    // 1/2(Ypredicted - Yexpected)^2
    fn cost_function(output_layer: &Layer, expected: f64) -> f64 {
        let error_sum = output_layer.get_neurons()
                                        .iter()
                                        .fold(0.0, |acc, neuron| acc + (0.5 * f64::powi(neuron.get_activation_value() - expected, 2)));

        error_sum
    }

    // derivative of cost function
    // dcost/dYpredicted = Ypredicted - Yexpected
    fn cost_function_derivative(output_layer: &Layer, expected: f64) -> f64 {
        let error_derivative_sum = output_layer.get_neurons()
                                                .iter()
                                                .fold(0.0, |acc, neuron| acc + ((neuron.get_activation_value() - expected) * Layer::sigmoid_derivative(neuron.get_activation_value())));
        error_derivative_sum
    }
}

#[cfg(test)] 
mod tests {
    use super::*;

    #[test]
    fn neural_net_initialization_check() {
        let neural_net = NeuralNet::new(3, 2, 5, 1, 0.1);

        assert_eq!(3, neural_net.layers.len());
        //input
        assert_eq!(2, neural_net.layers[0].get_neurons().len());
        //hidden
        assert_eq!(5, neural_net.layers[1].get_neurons().len());
        //output
        assert_eq!(1, neural_net.layers[2].get_neurons().len());
        assert_eq!(0.1, neural_net.learning_rate);

        //input
        assert_eq!(5, neural_net.layers[0].get_neurons()[0].get_weights().len());
        //hidden
        assert_eq!(1, neural_net.layers[1].get_neurons()[0].get_weights().len());
        //output
        assert_eq!(0, neural_net.layers[2].get_neurons()[0].get_weights().len());
    }

    #[test]
    fn cost_function_test() {
        let mut output_layer = Layer::new(2, 0);
        output_layer.set_default_neuron_values(0.5);
        let expected = 1.0;
        let cost = NeuralNet::cost_function(&output_layer, expected);

        assert_eq!(0.25, cost);
        output_layer.set_default_neuron_values(1.0);
        let expected = 4.0;
        let cost = NeuralNet::cost_function(&output_layer, expected);
        
        assert_eq!(9.0, cost);
    }

    fn cost_function_derivative_test() {
        let mut output_layer = Layer::new(2, 0);
        output_layer.set_default_neuron_values(0.5);
        let expected = 1.0;
        let cost = NeuralNet::cost_function_derivative(&output_layer, expected);

        assert_eq!(-1.0, cost);

        output_layer.set_default_neuron_values(7.0);
        let expected = 3.0;
        let cost = NeuralNet::cost_function_derivative(&output_layer, expected);

        assert_eq!(8.0, cost);
    }
}