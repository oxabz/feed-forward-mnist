use tch::nn::{self, *};
use tch::{index::*, Tensor};

use crate::ff_module::FFModule;

#[derive(Debug)]
pub(crate) struct MLP {
  hidden_layers: Vec<(Linear, Option<LayerNorm>)>,
  final_layer: Linear,
}

impl nn::Module for MLP {
  fn forward(&self, xs: &Tensor) -> Tensor {
    let mut xs = xs.shallow_clone();
    for (lin, norm) in &self.hidden_layers {
      xs = lin.forward(&xs);
      if let Some(normalization) = norm {
        xs = normalization.forward(&xs);
      }
      xs = xs.relu();
    }
    self.final_layer.forward(&xs)
  }
}

impl FFModule for MLP {
  fn forward_forward(&self, xs: &Tensor, layer: usize) -> Tensor {
    assert!(
      layer < self.hidden_layers.len(),
      "layer {} is out of bounds",
      layer
    );
    let layer = layer as i64;
    let mut xs = xs.shallow_clone();
    {
      let _guard = tch::no_grad_guard();
      for i in 0..layer {
        let (lin, norm) = &self.hidden_layers[i as usize];

        if let Some(normalization) = norm {
          xs = normalization.forward(&xs);
        }
        xs = lin.forward(&xs);
        xs = xs.relu();
      }
    }
    let (lin, norm) = &self.hidden_layers[layer as usize];
    if let Some(normalization) = norm {
      xs = normalization.forward(&xs);
    }
    xs = lin.forward(&xs);
    xs.relu()
  }

  fn layer_count(&self) -> usize {
    self.hidden_layers.len()
  }
}

pub(crate) struct MLPBuilder<'v> {
  path: &'v Path<'v>,
  hidden_layers: Vec<usize>,
  in_features: usize,
  out_features: usize,
  norm: bool,
}

impl<'v> MLPBuilder<'v> {
  pub fn new(path: &'v Path<'v>, in_features: usize, out_features: usize) -> MLPBuilder {
    MLPBuilder {
      path,
      hidden_layers: vec![],
      in_features,
      out_features,
      norm: false,
    }
  }

  pub fn add_hidden_layer(mut self, size: usize) -> MLPBuilder<'v> {
    self.hidden_layers.push(size);
    self
  }

  pub fn with_instance_norm(mut self) -> MLPBuilder<'v> {
    self.norm = true;
    self
  }

  pub fn build(self) -> MLP {
    let MLPBuilder {
      path,
      hidden_layers,
      in_features,
      out_features,
      norm,
    } = self;
    let hidden_layers_size = hidden_layers;

    let mut hidden_layers = vec![];
    let mut in_features = in_features;
    for (i, size) in hidden_layers_size.into_iter().enumerate() {
      let layer = {
        let path = path / format!("hidden_layer_{}", i);
        nn::linear(path, in_features as i64, size as i64, Default::default())
      };
      let norm = if norm {
        Some(nn::layer_norm(
          path / "norm",
          vec![size as i64],
          Default::default(),
        ))
      } else {
        None
      };

      hidden_layers.push((layer, norm));
      in_features = size;
    }

    let final_layer = nn::linear(
      path / "final_layer",
      in_features as i64,
      out_features as i64,
      Default::default(),
    );

    MLP {
      hidden_layers,
      final_layer,
    }
  }
}
