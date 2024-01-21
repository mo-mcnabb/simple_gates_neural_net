use rand::prelude::*;
pub struct Neuron {
    //TODO: look into making these values Option<f64>
    incoming_value: f64,
    activation_value: f64,
    weights: Vec<f64>
}

impl Neuron {
    pub fn new(num_weights: usize) -> Neuron {
        let mut rng = rand::thread_rng();
        let init_weights: Vec<f64> = (0..num_weights).map(|_| rng.gen()).collect();    
        
        Neuron {
            incoming_value: 0.0,
            activation_value: 0.0,
            weights: init_weights,
        }
    }

    pub fn get_incoming_value(&self) -> f64 {
        self.incoming_value
    }

    pub fn get_activation_value(&self) -> f64 {
        self.activation_value
    }

    pub fn get_weights(&self) -> Vec<f64> {
        self.weights.clone()
    }

    pub fn set_incoming_value(&mut self, incoming_value: f64) {
        self.incoming_value = incoming_value;
    }

    pub fn set_activation_value(&mut self, activation_value: f64) {
        self.activation_value = activation_value;
    }

    pub fn set_weights(&mut self, new_weights: Vec<f64>) {
        self.weights = new_weights;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_incoming_value_test() {
        let neuron = Neuron {
            incoming_value: 4.9,
            activation_value: 0.0,
            weights: Vec::new()
        };

        assert_eq!(4.9, neuron.get_incoming_value());
    }

    #[test]
    fn set_incoming_value_test() {
        let mut neuron = Neuron {
            incoming_value: 4.9,
            activation_value: 0.0,
            weights: Vec::new()
        }; 

        assert_eq!(4.9, neuron.get_incoming_value());

        neuron.set_incoming_value(7.8);
        assert_eq!(7.8, neuron.get_incoming_value());
    }

    #[test]
    fn get_activation_value_test() {
        let neuron = Neuron {
            incoming_value: 3.2,
            activation_value: 19.2,
            weights: Vec::new()
        };

        assert_eq!(19.2, neuron.get_activation_value());
    }

    #[test]
    fn set_activation_value_test() {
        let mut neuron = Neuron {
            incoming_value: 9.5,
            activation_value: 4.1,
            weights: Vec::new(),
        };

        assert_eq!(4.1, neuron.get_activation_value());

        neuron.set_activation_value(8.9);

        assert_eq!(8.9, neuron.get_activation_value());
    }

    #[test]
    fn set_weights_check_weights() {
        let mut neuron = Neuron {
            incoming_value: 0.0,
            activation_value: 0.0,
            weights: Vec::new(),
        };

        let new_weights = vec![1.0, 2.0, 3.0];

        neuron.set_weights(new_weights);
        assert_eq!(3, neuron.get_weights().len());
        assert_eq!(vec![1.0, 2.0, 3.0], neuron.get_weights());
    }

    #[test]
    fn new_neuron_weight_num_check() {
        let neuron = Neuron::new(72);

        assert_eq!(72, neuron.get_weights().len());

        let neuron = Neuron::new(1);

        assert_eq!(1, neuron.get_weights().len());

        let neuron = Neuron::new(0);

        assert_eq!(0, neuron.get_weights().len());
    }

    #[test]
    fn new_neuron_values_check() {
        let neuron = Neuron::new(4);

        assert_eq!(0.0, neuron.get_activation_value());
        assert_eq!(0.0, neuron.get_incoming_value());
    }
}