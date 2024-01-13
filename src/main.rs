// just for fun, learning both rust and neural networks

use rand::prelude::*;

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

    let mut rng = rand::thread_rng();
    let w1: f64 = rng.gen();
    let w2: f64 = rng.gen();
    let b: f64 = rng.gen();
    let learning_rate = 0.1;

    let (mut finished_w1, mut finished_w2, mut finished_b) = train(&and_set, (w1, w2, b), learning_rate);
    

    println!("-------------------AND SET---------------");
    for (x, y, z) in and_set.into_iter() {
        let sum = (x as f64 * finished_w1) + (y as f64 * finished_w2) + finished_b;
        let output = sigmoid(sum);

        println!("output: {output}, expected: {z}");
    }

    (finished_w1, finished_w2, finished_b) = train(&or_set, (w1, w2, b), learning_rate);

    println!("-------------------OR SET---------------");
    for (x, y, z) in or_set.into_iter() {
        let sum = (x as f64 * finished_w1) + (y as f64 * finished_w2) + finished_b;
        let output = sigmoid(sum);

        println!("output: {output}, expected: {z}");
    }

    (finished_w1, finished_w2, finished_b) = train(&nand_set, (w1, w2, b), learning_rate);
    println!("-------------------NAND SET---------------");
    for (x, y, z) in nand_set.into_iter() {
        let sum = (x as f64 * finished_w1) + (y as f64 * finished_w2) + finished_b;
        let output = sigmoid(sum);

        println!("output: {output}, expected: {z}");
    }

}

fn train(set: &Vec<(i32, i32, i32)>, weights_and_bias: (f64, f64, f64), learning_rate: f64) -> (f64, f64, f64) {
    
    let (mut w1, mut w2, mut b) = weights_and_bias;

    for _ in 0..=10_000_000{

        for i in 0..set.len() {
            let (x, y, answer) = set.get(i).unwrap();
            let sum = (*x as f64 * w1) + (*y as f64 * w2) + b;
            let output = sigmoid(sum);

            let error = *answer as f64 - output;
            

            let weight1_gradient = *x as f64 * error * (output * (1 as f64 - output));
            let weight2_gradient = *y as f64 * error * (output * (1 as f64 - output));
            let bias_gradient = error * (output * (1 as f64 - output));

            w1 = w1 + (learning_rate * weight1_gradient);
            w2 = w2 + (learning_rate * weight2_gradient);
            b = b + (learning_rate * bias_gradient);
        }
    }

    (w1, w2, b)
}

fn sigmoid(sum: f64) -> f64 {
    1.0 / (1.0 + (-sum).exp())
}
