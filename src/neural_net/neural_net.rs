pub struct NeuralNet {
    inputs: Vec<(i32, i32, i32)>,
    set_name: &str,
    input_layer: Layer,
    hidden_layers: Vec<Layer>,
    output_layer: Layer,
    bias: f64
}

impl NeuralNet {
    pub fn new(inputs: Vec<(i32, i32, i32)>, set_name: &str, num_hidden_layers: usize, num_hidden_layer_neurons: usize, num_outputs: usize, bias: f64) -> NeuralNet {
        
        let input_layer = Layer::new(2);
        let hidden_layers = (0..num_hidden_layers).map(|_| Layer::new(num_hidden_layer_neurons)).collect();
        let output_layer = Layer::new(num_outputs);

        NeuralNet {
            inputs: inputs,
            set_name: set_name,
            input_layer: input_layer,
            hidden_layers: hidden_layers,
            output_layer = output_layer,
            bias: bias
        }
    }
    
    pub fn train(&mut self, iterations: usize) {
        let inputs = &self.inputs;

        for _ in 0..iterations {
            for (x, y, answer) in inputs {

            }
        }

    }
}