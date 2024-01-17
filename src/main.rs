// just for fun, learning both rust and neural networks

mod perceptron;
mod neural_net;
fn main() {
    let and_set = vec![
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (1, 1, 1),
    ];

    let or_set = vec![
        (0, 0, 0),
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 1),
    ];

    let nand_set = vec![
        (0, 0, 1),
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 0),
    ];
/* 
    //0.1 is learning rate
    let mut perceptron = Perceptron::new(0.1);

    perceptron.train(&and_set, 1000);
    perceptron.test(&and_set, "AND SET");

    let mut perceptron = Perceptron::new(0.1);

    perceptron.train(&or_set, 1000);
    perceptron.test(&or_set, "OR SET");

    let mut perceptron = Perceptron::new(0.1);
    
    perceptron.train(&nand_set, 1000);
    perceptron.test(&nand_set, "NAND SET");
    */
}

    