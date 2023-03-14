use std::f64::consts::E;

use tch::{nn::VarStore, Tensor};

/**
 * A forward-forward optimizer.
 */
pub trait ForwardForwardOptimizer {
  /**
   * Performs a partial step of the optimizer.
   * It get the learning rate compute the backword pass and update the weights.
   */
  fn partial_step(&self, badness: &Tensor, depth: usize);

  fn step(&mut self);

  fn learning_rate(&self) -> f64;

  fn zero_grad(&self);

  fn current_step(&self) -> u32;

  fn set_step(&mut self, step: u32);
}

/**
 * A forward-forward optimizer that uses a slow start and then exponential decay.
 *
 * # Learning Rate
 *
 * The learing rate follows the formula:
 * ax² + e^-bx + c
 * where :
 * b = 2 / ramp_up_step
 * c = base_lr
 * a = (max_lr - base_lr) / (ramp_up_step ^ 2 * e ^ -(b * ramp_up_step))
 */
pub struct ForwardForwardSlowStartExponentialDecay<'v> {
  base_lr: f64,
  max_lr: f64,
  ramp_up_step: u32,
  a: f64,
  b: f64,
  c: f64,
  vs: &'v VarStore,
  step: u32,
}

impl<'v> ForwardForwardSlowStartExponentialDecay<'v> {
  pub fn new(base_lr: f64, max_lr: f64, ramp_up_step: u32, vs: &'v VarStore) -> Self {
    let b = 2.0 / ramp_up_step as f64;
    let c = base_lr;
    let a = (max_lr - base_lr) / ((ramp_up_step as f64).powi(2) * (-b * ramp_up_step as f64).exp());
    Self {
      base_lr,
      max_lr,
      ramp_up_step,
      a,
      b,
      c,
      vs,
      step: 0,
    }
  }

  // Getters

  pub fn base_lr(&self) -> f64 {
    self.base_lr
  }

  pub fn max_lr(&self) -> f64 {
    self.max_lr
  }

  pub fn ramp_up_step(&self) -> u32 {
    self.ramp_up_step
  }


  // Setters

  pub fn set_base_lr(&mut self, base_lr: f64) {
    self.base_lr = base_lr;
    self.c = base_lr;
    self.a = (self.max_lr - base_lr) / ((self.ramp_up_step as f64).powi(2) * (-self.b * self.ramp_up_step as f64).exp());
  }

  pub fn set_max_lr(&mut self, max_lr: f64) {
    self.max_lr = max_lr;
    self.a = (max_lr - self.base_lr) / ((self.ramp_up_step as f64).powi(2) * (-self.b * self.ramp_up_step as f64).exp());
  }

  pub fn set_ramp_up_step(&mut self, ramp_up_step: u32) {
    self.ramp_up_step = ramp_up_step;
    self.b = 2.0 / ramp_up_step as f64;
    self.a = (self.max_lr - self.base_lr) / ((ramp_up_step as f64).powi(2) * (-self.b * ramp_up_step as f64).exp());
  }
  
}

impl<'v> ForwardForwardOptimizer for ForwardForwardSlowStartExponentialDecay<'v> {
  fn step(&mut self) {
    self.step += 1;
  }

  fn learning_rate(&self) -> f64 {
    let s = self.step as f64;
    let a = self.a;
    let b = self.b;
    let c = self.c;

    a * s.powi(2) * E.powf(-b * s) + c
  }

  fn zero_grad(&self) {
    self.vs.variables().iter_mut().for_each(|(_, v)| v.zero_grad());
  }

  fn partial_step(&self, badness: &Tensor, _depth: usize) {
    let lr = self.learning_rate();
    self.zero_grad();
    badness.backward();
    self.vs.variables().iter_mut().for_each(|(_,  v)| {
      let _guard = tch::no_grad_guard();
      let grad = v.grad();

      if !grad.defined() {
        return;
      }
      *v -= grad * lr;
    });

  }

  // Getter 

  fn current_step(&self) -> u32 {
    self.step
  }

  // Setter

  fn set_step(&mut self, step: u32) {
    self.step = step;
  }
}
