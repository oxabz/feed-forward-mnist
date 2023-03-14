#![feature(trait_alias)]

mod ff_mlp;
mod ff_module;
mod ff_optimizer;
mod mlp;
mod negatif_generator;
mod utils;

use ff_mlp::{FFMLP};
use tch::{nn::{self, Adam, OptimizerConfig}, index::*};

use crate::{negatif_generator::forward_forward_mnist, ff_module::ModuleTY};

fn main() {
  let device = tch::Device::Cpu;
  let batch_size = 512;
  let epochs = 100u32;
  // Loading the data
  let ff_ds = forward_forward_mnist();

  // Creating the loss function
  let threshold = 2.0;
  let sharpness = 1.0;
  let layer_loss = |xs: &tch::Tensor, ys: &tch::Tensor| {
    let ys = ys.clamp(0.0, 1.0);
    let ys = ys - 0.5;
    let ys = ys * -2.0;

    let xs = xs.pow_tensor_scalar(2.0).mean_dim(vec![-1].as_slice(), false, tch::Kind::Float);
    let loss = (xs - threshold) * ys;
    let loss = (loss.exp() * sharpness + 1.0).log().mean(tch::Kind::Float);
    loss
  };

  // Creating the model
  let vs = nn::VarStore::new(device);
  let optim = Adam::default();
  let optim = || optim.clone().build(&vs, 1e-3).unwrap();
  let mlp = FFMLP::new(&vs.root(), 28 * 28, &[2000, 2000, 2000, 2000], 10, layer_loss, optim);
  let mut optim = optim();

  // Training the model through forward-forward
  for epoch in 0..epochs {
    let mut losses = vec![];
    for (xs, ys) in ff_ds.train_iter(batch_size) {
      let xs = xs.view([-1, 28 * 28]);
      let ys = ys.view([-1, 1]);
      let pred = mlp.forward_ty(&xs, &ys);
      let pred = pred.sigmoid();
      
      let ys = ys.view([-1]);
      let loss = pred.cross_entropy_for_logits(&(&ys - 1.0).clamp(0.0, 12.0).to_kind(tch::Kind::Int64)) * ys.clamp(0.0, 1.0).mean(tch::Kind::Float);
      {
        optim.backward_step(&loss);
      }
      losses.push(f32::from(loss));
    }
    println!("Epoch: {:4}, Loss: {:8.4}", epoch, losses.iter().sum::<f32>() / losses.len() as f32);
  }
}
