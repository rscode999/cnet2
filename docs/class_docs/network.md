# Network

Trainable predictor with user-defined structure.

### Constructor

*Signature:* `Network()`

Creates a new, empty network.

The new network has no components, loss calculator, or optimizer.

---

### Getters

#### active_branch_indices

*Signature:* `std::vector<int32_t> active_branch_indices() const`

Returns the 0-based indices of the ends of each branch in the internal component storage.

The output's length is equal to the total number of branches used in the network so far (but the branches may not necessarily still exist).

Index `i` equals the constant `NETWORK_BRANCH_COMBINED` (a negative value) if branch `i` has been combined with another branch, and thus no longer exists.

**Returns**

* `std::vector<int32_t>`: Indices of leaf nodes.

---

#### is_enabled

*Signature:* `bool is_enabled() const`

Returns whether the network is ready for training and optimization.

**Returns**

* `bool`: Whether the network is enabled.

---

### Setters

#### add_combiner

*Signature:* `void add_combiner(std::initializer_list<int32_t> branch_ids_to_combine, int32_t branch_id = 0, std::source_location loc = std::source_location::current())`

Adds a combiner, merging the branch IDs given in `branch_ids_to_combine`, to branch `branch_id`.

If `branch_id` is negative, at least the total number of branches used so far, or corresponds to a branch that has been merged, this method throws `cast::bad_component_addition`.

`cast::bad_component_addition` is also thrown if any of the branch IDs in `branch_ids_to_combine` is out of the range [0, <number of branches used in the network - 1>], has already been merged, or equals `branch_id` (combiners cannot merge their own branch).

A Combiner cannot be the first component added to a network.

To use this method, the network cannot be enabled. 

**Parameters**

* `branch_ids_to_combine` (`std::initializer_list<int32_t>`): List of branch IDs to merge. Non-empty.
* `branch_id` (`int32_t`): Branch to add the new combiner to.
* `loc` (`std::source_location`): Location where this method is called (for debugging purposes).

**Exceptions**

* `cast::bad_network_config`: Thrown if the network is enabled.
* `cast::bad_component_addition`: Thrown if `branch_id` is negative, out of range, or already combined; if any branch ID to combine is out of range, already combined, or equals `branch_id`; or if attempting to add a combiner as the first component.
* `std::out_of_range`: If more than 2 billion components (operators, splitters, or combiners) have been added to the network.

---

#### add_operator

*Signature:* `void add_operator(std::shared_ptr<Operator> op, int32_t branch_id = 0, std::source_location loc = std::source_location::current())`

Adds `op` to the end of branch `branch_id`.

An operator is a layer or an activation function.

To use this method, the network cannot be enabled.

**Parameters**

* `op` (`std::shared_ptr<Operator>`): Operator to add to a branch.
* `branch_id` (`int32_t`): Branch to add the new operator to.
* `loc` (`std::source_location`): Location where this method is called (for debugging purposes).

**Exceptions**

* `cast::bad_network_config`: If the network is enabled.
* `cast::bad_component_addition`: If `branch_id` is negative, out of range, or corresponds to a merged branch.
* `std::out_of_range`: If more than 2 billion components (operators, splitters, or combiners) have been added to the network.

---

#### add_splitter

*Signature:* `void add_splitter(int32_t branch_count, int32_t branch_id = 0, std::source_location loc = std::source_location::current())`

Adds a splitter that distributes execution across `branch_count` new branches, to branch `branch_id`.

To use this method, the network cannot be enabled.

**Parameters**

* `branch_count` (`int32_t`): Number of branches to split execution into. Must be at least 2.
* `branch_id` (`int32_t`): Branch to add the new splitter to.
* `loc` (`std::source_location`): Location where this method is called (for debugging purposes).

**Exceptions**

* `cast::bad_network_config`: If the network is enabled.
* `cast::bad_component_addition`: If `branch_id` is negative, out of range, or corresponds to a merged branch.
* `std::out_of_range`: If more than 2 billion components (operators, splitters, or combiners) have been added to the network.

---

#### disable

*Signature:* `void disable()`

Disables the network.

Prevents training and optimization, but allows more components to be added.

---

#### enable

*Signature:* `void enable()`

Checks if the network has the necessary components to run.

If not, throws `enable_failed_error`. If so, allows training and optimization.

Enable check:  
The network must have a loss calculator, optimizer, at least one component, exactly one output, and a valid first component.

**Exceptions**

* `cast::enable_failed_error`: If the enable check fails.

---

#### set_loss_calculator

*Signature:* `void set_loss_calculator(std::shared_ptr<LossCalculator> calc)`

Sets this network's loss calculator to `calc`.

To use this method, the network cannot be enabled.

**Parameters**

* `calc` (`std::shared_ptr<LossCalculator>`): New loss calculator to use. Not equal to `nullptr`.

**Exceptions**

* `cast::bad_network_config`: If the network is enabled.

---

#### set_optimizer

*Signature:* `void set_optimizer(std::shared_ptr<Optimizer> optim)`

Sets this network's optimizer to `optim`.

The pointer to the optimizer can be manipulated from outside the network.

To use this method, the network cannot be enabled.

**Parameters**

* `optim` (`std::shared_ptr<Optimizer>`): New optimizer to use. Not equal to `nullptr`.

**Exceptions**

* `cast::bad_network_config`: If the network is enabled.

---

#### set_optimizer_hyperparams

*Signature:* `void set_optimizer_hyperparams(std::initializer_list<double> new_hyperparams)`

Sets the hyperparameters of the network's optimizer.

The length and preconditions of each element in `new_hyperparams` must match those
in the optimizer's `set_hyperparameters` method.

**Parameters**

* `new_hyperparams` (`std::initializer_list<double>`): New hyperparameters to set.

**Exceptions**

* `cast::bad_network_config`: Thrown if the network has no optimizer.

---

### Methods

#### forward

*Signature:* `xt::xarray<double> forward(xt::xarray<double> input)`

Returns the result of the network's forward pass on `input`.

To use this method, the network must be enabled.

**Parameters**

* `input` (`xt::xarray<double>`): Tensor to compute forward pass on.

**Returns**

* `xt::xarray<double>`: Result of the forward pass.

**Exceptions**

* `cast::bad_network_config`: If the network is not enabled.
* `cast::shape_error`: If input shapes are incompatible during computation.
* `std::runtime_error`: If the forward pass finishes without reaching an output node.

---

#### backward

*Signature:* `void backward(xt::xarray<double> predicted, xt::xarray<double> expected)`

Computes the backward pass, beginning with loss between `predicted` and `expected`.

Stores updated gradients inside the network layers, for use by the network's optimizer.

The network must be enabled to use this method.

**Parameters**

* `predicted` (`xt::xarray<double>`): Network's prediction for a given input.
* `expected` (`xt::xarray<double>`): What the network should have predicted instead of `predicted`.

**Exceptions**

* `bad_network_config`: If the network is not enabled.

---

#### optimize

*Signature:* `void optimize(bool zero_grad = true)`

Runs an optimization pass on the network's layers.

Uses the network's stored optimizer and the gradients computed from the `backward` method.

To use this method, the network must be enabled.

**Parameters**

* `zero_grad` (`bool`): Whether to set all operator's gradients to 0 after computing the optimization pass.

**Exceptions**

* `cast::bad_network_config`: If the network is not enabled.

---

### Operator Overloads

#### output stream insertion (<<)

*Signature:* `template<typename CharT, typename Traits> friend std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& output_stream, const Network& network)`

Exports `network` to the output stream `output_stream`, returning `output_stream` with `network`'s information inside.

Works for any output stream, including `std::wcout`, the wide-character output.

**Parameters**

* `output_stream` (`std::basic_ostream<CharT, Traits>&`): Stream to put the network into.
* `network` (`const Network&`): Network object to export.

**Returns**

* `std::basic_ostream<CharT, Traits>&`: `output_stream` with `network` inserted.