// just for fun, learning both rust and neural networks
use rand::prelude::*;
mod perceptron;
mod neural_net;

fn main() {
    let and_set = vec![
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (1, 1, 1),
    ];

    let _or_set = vec![
        (0, 0, 0),
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 1),
    ];

    let _nand_set = vec![
        (0, 0, 1),
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 0),
    ];

    let test_set: Vec<(f64, f64, f64)> = vec![
       (0.5, 0.6, 0.7),
    ];
    let mut rng = rand::thread_rng();

    let test: f64 = 23.1347234523452345234523452345;

    let scale: f64 = 1000.0;
    let p: f64 = (test * scale).round() / scale;
    println!("{p}");
}

    