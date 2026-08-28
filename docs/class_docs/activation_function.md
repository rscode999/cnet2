# ActivationFunction

Network component that computes an element-wise function for all of its inputs.

All activation function objects have constructors that take no parameters.

Abstract class. Subclass of `Operator`.

## Shared Virtual Methods

### Constructor

#### ActivationFunction

*Signature:* `<ActivationFunction concrete subclass>()`

Creates a new activation function object

### Methods

#### compute

*Signature:* `virtual std::vector<xt::xarray<double>> compute(std::vector<xt::xarray<double>> inputs) = 0`

Computes the element-wise activation function applied to each parameter in `inputs`.

**Parameters**

* `inputs` (`std::vector<xt::xarray<double>>`): List of values to compute. Non-empty.

**Returns**

* `std::vector<xt::xarray<double>>`: Computed activation values for each element of `inputs`.

#### compute_backwards_pass

*Signature:* `virtual std::vector<xt::xarray<double>> compute_backwards_pass(std::vector<xt::xarray<double>> upstream_gradients) = 0`

Computes the derivative of the activation function applied to each parameter of `upstream_gradients`.

**Parameters**

* `upstream_gradients` (`std::vector<xt::xarray<double>>`): List of values to compute. Non-empty.

**Returns**

* `std::vector<xt::xarray<double>>`: Derivative values for each element of `upstream_gradients`.

---
---
---

## Sigmoid

For a number x, the Sigmoid function is given by
$sigmoid(x) = \frac{1}{1 + e^-x}$.

The result is always between 0 and 1, exclusive on both ends.

The result of a Sigmoid's `to_string` method is "sigmoid".

---
---
---

## ReLU

Rectified Linear Unit (ReLU) function.

For a number x, the ReLU function is given by
$ReLU(x) = max(0, x)$.

The result of a ReLU's `to_string` method is "relu".