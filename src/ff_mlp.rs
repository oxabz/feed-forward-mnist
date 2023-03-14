use std::sync::atomic::AtomicU32;
use std::sync::{Mutex};

use derivative::Derivative;
use tch::{nn::*, Tensor};

use crate::ff_module::ModuleTY;

pub trait LossFn = Fn(&Tensor, &Tensor) -> Tensor + Send + Sync + Copy;

pub fn fmt_loss_fn<F: LossFn>(_: &F, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
  write!(fmt, "LossFn")
}

#[derive(Derivative)]
#[derivative(Debug)]
pub struct FFMLPLayer<F>
where
  F: LossFn,
{
  lin: Linear,
  norm: LayerNorm,
  #[derivative(Debug(format_with = "fmt_loss_fn", bound = ""))]
  loss_fn: F,
  optim: Mutex<Optimizer>,
  c: AtomicU32,
}

impl<F: LossFn> FFMLPLayer<F> {
  pub fn new(
    vs: &Path,
    input_size: i64,
    output_size: i64,
    loss_fn: F,
    optim: Optimizer,
  ) -> Self {
    FFMLPLayer {
      lin: linear(vs / "lin", input_size, output_size, Default::default()),
      norm: layer_norm(
        vs / "norm",
        vec![input_size],
        LayerNormConfig {
          elementwise_affine: false,
          ..Default::default()
        },
      ),
      loss_fn,
      optim: Mutex::new(optim),
      c: AtomicU32::new(0),
    }
  }
}

impl<F: LossFn> Module for FFMLPLayer<F> {
  fn forward(&self, xs: &Tensor) -> Tensor {
    let xs = xs / (xs.linalg_norm(2, vec![1].as_slice(), true, tch::Kind::Float) + 1e-5);
    let xs = self.lin.forward(&xs);
    xs.relu()
  }
}

impl<F: LossFn> ModuleTY for FFMLPLayer<F> {
  fn forward_ty(&self, xs: &Tensor, ys: &Tensor) -> Tensor {
    // Compute the loss
    let loss = {
      let xs = self.forward(xs);
      (self.loss_fn)(&xs, ys)
    };

    // Update the gradients
    self.optim.lock().unwrap().backward_step(&loss);

    let c =self.c.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if c % 100 == 0 {
      println!("Loss: {}", f32::from(loss));
    }

    // Recompute the forward pass without gradients
    let _guard = tch::no_grad_guard();
    let xs = self.forward(xs);
    xs
  }
}

#[derive(Derivative)]
#[derivative(Debug(bound = ""))]
pub struct FFMLP<F>
where
  F: LossFn,
{
  hidden_layers: Vec<FFMLPLayer<F>>,
  final_layer: Linear,
}

impl <F:LossFn>FFMLP<F> {
  pub fn new<FO:Fn()->Optimizer>(vs: &Path, input_size: i64, hidden_sizes: &[i64], output_size: i64, loss_fn:F, optim: FO) -> Self {
    let mut hidden_layers = Vec::new();
    let mut prev_size = input_size;
    for &hidden_size in hidden_sizes {
      hidden_layers.push(FFMLPLayer::new(
        &(vs / hidden_layers.len()),
        prev_size,
        hidden_size,
        loss_fn,
        optim()
      ));
      prev_size = hidden_size;
    }
    let final_layer = linear(
      vs / "final_layer",
      prev_size,
      output_size,
      Default::default(),
    );
    FFMLP {
      hidden_layers,
      final_layer,
    }
  }
}

impl <F:LossFn>Module for FFMLP<F> {
  fn forward(&self, xs: &Tensor) -> Tensor {
    let mut xs = xs.shallow_clone();
    self.hidden_layers.iter().for_each(|layer| {
      xs = layer.forward(&xs);
    });
    self.final_layer.forward(&xs)
  }
}

impl <F:LossFn>ModuleTY for FFMLP<F> {
  fn forward_ty(&self, xs: &Tensor, ys: &Tensor) -> Tensor {
    let mut xs = xs.shallow_clone();
    self.hidden_layers.iter().for_each(|layer| {
      xs = layer.forward_ty(&xs, ys);
    });
    self.final_layer.forward(&xs)
  }
}
