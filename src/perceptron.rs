use rand::prelude::*;

pub struct Perceptron {
    w1: f64,
    w2: f64,
    b: f64,
    learning_rate: f64,
}

impl Perceptron {
    pub fn new(learning_rate: f64) -> Perceptron {
        let mut rng = rand::thread_rng();
        let w1: f64 = rng.gen();
        let w2: f64 = rng.gen();
        let b: f64 = rng.gen();

        Perceptron {
            w1: w1,
            w2: w2,
            b: b,
            learning_rate: learning_rate,
        }
    }
    
    pub fn train(&mut self, set: &Vec<(i32, i32, i32)>) {
        
        for _ in 0..=10_000 {

            for i in 0..set.len() {
                let (x, y, answer) = set.get(i).unwrap();
                let sum = (*x as f64 * self.w1) + (*y as f64 * self.w2) + self.b;
                let output = Self::sigmoid(sum);
    
                let error = *answer as f64 - output;
                
                let sigmoid_derivative = Self::sigmoid_derivative(output);
    
                let weight1_gradient = *x as f64 * error * sigmoid_derivative;
                let weight2_gradient = *y as f64 * error * sigmoid_derivative;
                let bias_gradient = error * sigmoid_derivative;
    
                self.w1 = self.w1 + (self.learning_rate * weight1_gradient);
                self.w2 = self.w2 + (self.learning_rate * weight2_gradient);
                self.b = self.b + (self.learning_rate * bias_gradient);
            }
        }
    }

    pub fn test(&self, set: &Vec<(i32, i32, i32)>, set_name: &str) {
        println!("-------------------{set_name}---------------");
        for (x, y, answer) in set.into_iter() {
            let sum = (*x as f64 * self.w1) + (*y as f64 * self.w2) + self.b;
            let output = Self::sigmoid(sum);

            println!("output: {output}, expected: {answer}");
        }
    }

    fn sigmoid(sum: f64) -> f64 {
        1.0 / (1.0 + (-sum).exp())
    }

    fn sigmoid_derivative(output: f64) -> f64 {
        output * (1 as f64 - output)
    }
}