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

    pub fn sigmoid(sum: f64) -> f64 {
       1.0 / (1.0 + (-sum).exp())
    }

    pub fn sigmoid_derivative(input: f64) -> f64 {
        input * (1.0 - input)
    }

    pub fn feed_forward(&mut self, next_layer: &mut Layer) {

        next_layer.apply_to_neurons(|index, neuron| {

            let sum = self.neurons.iter()
                .fold(0.0, |acc, neuron| acc + (neuron.get_weights()[index] * neuron.get_activation_value()));

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


    fn truncate_to_3_decimal_places(val: f64) -> f64 {
        let scale = 1000.0;
        (val * scale).round() / scale
    }
}


