pub struct Layer {
    //neurons: Vec<Neurons>
}

impl Layer {
    pub fn new() -> Layer {
        Layer {

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