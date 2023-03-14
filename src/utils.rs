use tch::Tensor;

/**
 * Given the activation of a hidden layer, returns the prediction of the layer on wether the input is positive or not.
 * 
 * # Formula 
 * 
 * Formula 1 of the paper ["The Forward-Forward Algorithm: Some Preliminary Investigations"](https://arxiv.org/pdf/2212.13345.pdf)
 * 
 * # Arguments
 * - `activation`: the activation of the hidden layer (a tensor of shape [batch_size, hidden_size])
 * - `threashold`: the threashold of activation to consider the input as positivea 
 * 
 * # Returns
 * - Tensor : a tensor of shape [batch_size, 1] containing the prediction of the layer on wether the input is positive or not.
 */
pub fn positive_pred(activation: &Tensor, threashold: f64) -> Tensor {
  (activation.pow_tensor_scalar(2.0)
    .sum_dim_intlist(vec![-1].as_slice(), true, tch::Kind::Float) 
    - threashold).sigmoid()
}

/**
 * Given the activation of a hidden layer and the ground truth. Returns the goodness score of the layer.
 * The mean goodness is the mean square error between the prediction of the layer and the ground truth.
 *  
 * # Arguments
 * - `activation`: the activation of the hidden layer (a tensor of shape [batch_size, hidden_size])
 * - `threashold`: the threashold of activation to consider the input as positivea
 * - `y`: the ground truth (a tensor of shape [batch_size, 1])
 * 
 */
pub fn mean_goodness(activation: &Tensor, threashold: f64, y: &Tensor) -> Tensor {
  let pred = positive_pred(activation, threashold);
  let diff = pred - y;
  diff.pow_tensor_scalar(2.0).mean(tch::Kind::Float)
}

 