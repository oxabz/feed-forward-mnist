use tch::Tensor;

// pub trait FFModule<I> : tch::nn::Module
//   where I: Iterator<Item = Tensor> {
//   fn forward_forward(&self, xs: &Tensor) -> I;
// }

pub trait FFModule: tch::nn::Module {
  fn forward_forward(&self, xs: &Tensor, layer:usize) -> Tensor;

  fn layer_count(&self) -> usize;
}

/**
 * A trait for modules that behave differently during training and inference and need to know the labels to do so.
 */
pub trait ModuleTY: tch::nn::Module {
  /**
   * Forward pass for training.
   */
  fn forward_ty(&self, xs: &Tensor, ys:&Tensor) -> Tensor;
}
