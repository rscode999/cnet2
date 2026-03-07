#include "optimizer.cpp"
#include <stdexcept>

#pragma once

/**
 * Thrown to indicate that the network is not in the proper state to call a method.
 * 
 * Subclass of `std::runtime_error`.
 */
class illegal_state : public std::runtime_error {

public:

    /**
     * Creates a new exception with `error_message` as the error message
     * @param error_message the error message to display upon throw
     */
    illegal_state(const std::string& error_message) : runtime_error(error_message) {
    }

};





namespace CNet {


    

/**
 * A neural network that can be trained and used for predictions.
 * 
 * The user adds layers, a loss calculator, and an optimizer to the network prior to use.
 * The input layer is the first layer. The output layer is the last layer. Layers use 0-based indexing.
 * 
 * To use a network, the network must be enabled, by calling the `{networkName}.enable()` method.
 * `enable` checks if the network has valid settings, i.e. layer inputs and outputs are valid.
 * 
 * Components of a network can be changed at any time, provided that it is not enabled.
 * Disable a network with `{networkName}.disable()`.
 * 
 * Important Note: Softmax activations are not allowed, except in the network's final layer.
 */
class Network {


private:


    /**
     * Whether the network is enabled (ready for training and evaluation).
     * 
     * A network is operation-ready if the network has at least 1 layer,
     * a loss calculator and optimizer are defined,
     * and the input/output dimensions of each layer are compatible.
     * Also, no layer may have Softmax activation, except for the final layer.
     */
    bool enabled;


    /**
     * Stores the input vector given to the network
     * 
     * Used in backpropagation.
     */
    std::vector<Eigen::VectorXd> initial_inputs;

    /**
     * Stores outputs from before and after the activation function is applied at each layer.
     * 
     * Used in backpropagation.
     */
    std::vector<std::vector<LayerCache>> intermediate_outputs;


    /**
     * The network's (linear) layers.
     * Index 0 is the input. The final index is the output.
     */
    std::vector<Layer> layers;


    /**
     * Smart pointer to object that calculates losses
     */
    std::shared_ptr<LossCalculator> loss_calc;

    /**
     * Smart pointer to object that improves the model's weights
     */
    std::shared_ptr<Optimizer> optim;




public:

    /**
     * Creates an empty network.
     * 
     * The new network is not enabled. It has no layers, loss calculator, or optimizer.
     */
    Network() {
        enabled = false;
    }


    ////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////
    //GETTERS

    /**
     * Returns a deep copy of the bias vector in layer `layer_number`.
     * 
     * Layers use 0-based indexing. The first layer is at layer number 0.
     * 
     * @param layer_number layer number to access
     * @return biases of layer `layer_number`
     * @throws `std::out_of_range` if `layer_number` is not on the interval [0, `{network}.layer_count()` - 1]
     */
    Eigen::VectorXd biases_at(int layer_number) const {
        assert((layer_number>=0 && layer_number<(int)layers.size() && "Layer number for bias vector access must be on the interval [0, # layers - 1]"));
        return layers[layer_number].bias_vector();
    }



    /**
     * @return the number of inputs of this network
     * @throws `illegal_state` if the network has no layers
     */
    int input_dimension() const {
        if((int)layers.size()<1) {
            throw illegal_state("The network must have at least 1 layer to get input dimension");
        }
        return layers[0].input_dimension();
    }



    /**
     * @return whether the network is enabled
     */
    bool is_enabled() const {
        return enabled;
    }



    /**
     * @return deep copy of the layer at `layer_number` (0-based indexing). Must be between 0 and `{network}.layer_count()`-1
     */
    Layer layer_at(int layer_number) const {
        assert((layer_number>=0 && layer_number<layer_count() && "To retrieve a layer, layer number must be between 0 and (number of layers)-1, inclusive on both ends"));
        return layers[layer_number];
    }



    /**
     * Returns the layer whose name is `layer_name`.
     * 
     * The first matching layer name, i.e. the layer closest to the input layer, is returned.
     * 
     * If no layer with `layer_name` is found, throws `std::out_of_range`.
     * 
     * @param layer_name name of layer to find
     * @return layer with the lowest index whose name is `layer_name`
     * @throws `std::out_of_range` if no matching layer name is found
     */
    Layer layer_at(const std::string& layer_name) const {
        for(Layer l : layers) {
            if(l.name() == layer_name) {
                return l;
            }
        }

        throw std::out_of_range("Could not find any matching layers with the given name");
    }



    /**
     * @return the number of layers in the network
     */
    int layer_count() const {
        return (int)layers.size();
    }



    /**
     * @return smart pointer to the Network's loss calculator
     */
    std::shared_ptr<LossCalculator> loss_calculator() const {
        return this->loss_calc;
    }



    /**
     * @return smart pointer to the network's optimizer object
     * 
     * The returned pointer can be used to directly change the network's optimizer.
     */
    std::shared_ptr<Optimizer> optimizer() const {
        return optim;
    }



    /**
     * @return std::vector containing the optimizer's hyperparameters, in the order required by the current optimizer's `set_optimizer_hyperparameters` method
     */
    std::vector<double> optimizer_hyperparameters() const {
        return optim->hyperparameters();
    }


    /**
     * @return the number of outputs of this network
     * @throws `illegal_state` if the network has less than 1 layer
     */
    int output_dimension() const {
        if((int)layers.size()<1) {
            throw illegal_state("The network must have at least 1 layer");
        }
        return layers[(int)layers.size() - 1].output_dimension();
    }



    /**
     * Returns a deep copy of the weight matrix in layer `layer_number`.
     * 
     * Layers use 0-based indexing. The first layer is at layer number 0.
     * 
     * @param layer_number layer number to access. Must be between 0 and `{networkName}.layer_count()`-1, inclusive on both sides
     * @return weights of layer `layer_number`
     * @throws `std::out_of_range` if `layer_number` is not a valid index number
     */
    Eigen::MatrixXd weights_at(int layer_number) const {
        assert((layer_number>=0 && layer_number<(int)layers.size() && "Layer number must be between 0 and (number of layers)-1"));
        return layers[layer_number].weight_matrix();
    }



    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////
    //SETTERS (MOST MAY NOT BE CALLED IF THE NETWORK IS ENABLED)

    /**
     * Adds `new_layer` to the back of the network.
     * 
     * If the network is enabled, this method throws `illegal_state`.
     * 
     * @param new_layer layer to add
     * @throws `illegal_state` if the network is enabled
     */
    void add_layer(Layer new_layer) {
        //NO REFERENCES! The user can change the layers if so.

        if(enabled) {
            throw illegal_state("Network must not be enabled to add a layer");
        }
        layers.push_back(new_layer);
    }



    /**
     * Adds a new layer to the back of the network.
     * The new layer has `input_dimension` inputs, `output_dimension` outputs, and a name of `name`.
     * 
     * The weights and biases of the layer are initialized to random numbers 
     * on the interval [-`initialization_scale_factor`, `initialization_scale_factor`].
     * 
     * The layer does not have an activation function.
     * 
     * @param input_dimension dimension of the layer's input
     * @param output_dimension dimension of the layer's output
     * @param name name of the layer. Default: `"layer"`
     * @param initialization_scale_factor factor to multiply weights and biases by. Default 1.
     * @throws `illegal_state` if the network is enabled
     */
    void add_layer(int input_dimension, int output_dimension, std::string name="layer", int initialization_scale_factor = 1) {
        if(enabled) {
            throw illegal_state("Network must not be enabled to add a layer");
        }
        Layer new_layer = Layer(input_dimension, output_dimension, name, initialization_scale_factor);
        layers.push_back(new_layer);
    }



    /**
     * Adds a new layer to the back of the network.
     * The new layer has `input_dimension` inputs, `output_dimension` outputs, 
     * an activation function of `activation_function`. and a name of `name`.
     * 
     * The weights and biases of the layer are initialized to random numbers 
     * on the interval [-`initialization_scale_factor`, `initialization_scale_factor`].
     * 
     * @param input_dimension dimension of the layer's input
     * @param output_dimension dimension of the layer's output
     * @param activation_function smart pointer to activation function of the new layer
     * @param name name of the layer. Default: `"layer"`
     * @param initialization_scale_factor factor to multiply weights and biases by. Default 1.
     * @throws `illegal_state` if the network is enabled
     */
    void add_layer(int input_dimension, int output_dimension, std::shared_ptr<ActivationFunction> activation_function, std::string name="layer", int initialization_scale_factor = 1) {
        if(enabled) {
            throw illegal_state("Network must not be enabled to add a layer");
        }
        Layer new_layer = Layer(input_dimension, output_dimension, activation_function, name, initialization_scale_factor);
        layers.push_back(new_layer);
    }



    /**
     * Disables the network, allowing the network to be edited.
     */
    void disable() {
        enabled = false;
    }



    /**
     * Enables the network, allowing training and predictions.
     * 
     * Before being enabled, this method performs a check.
     * 
     * Check: The network must have at least 1 layer, a loss calculator, and an optimizer.
     * The output dimension of each layer must equal the input dimension of the next layer.
     * The only layer that can have Softmax activation is the final (output) layer.
     * 
     * If the check fails, the method throws `illegal_state`.
     * 
     * If the check passes, to prepare for network architecture changes, all training state information is reset.
     * 
     * @throws `illegal_state` (with descriptive error message) if the network state check fails
     */
    void enable() {

        //Ensure there are at least 1 layer
        if((int)layers.size() < 1) {
            throw illegal_state("Enable check failed- The network needs at least 1 layer to begin training");
        }

        //Ensure there is a loss calculator
        if(!loss_calc) {
            throw illegal_state("Enable check failed- The network must have a loss calculator to begin training");
        }

        //Ensure there is an optimizer
        if(!optim) {
            throw illegal_state("Enable check failed- The network must have an optimizer to begin training");
        }

        //Check inputs and outputs of each layer. Also check layers for Softmax activation
        for(int i = 0; i < (int)layers.size() - 1; i++) {
            //Dimension compatibility
            if(layers[i].output_dimension() != layers[i+1].input_dimension()) {
                throw illegal_state("Enable check failed- Output dimension of layer " + std::to_string(i) + " (dimension=" + std::to_string(layers[i].output_dimension()) +
                ") must equal the input dimension of layer " + std::to_string(i+1) + " (dimension=" + std::to_string(layers[i+1].input_dimension()) + ")");
            }

            //Softmax activation in non-output layers (the output is the last layer)
            if(layers[i].activation_function()->name() == "softmax") {
                throw illegal_state("Enable check failed- Layer " + std::to_string(i) + " (\"" + layers[i].name() + "\") is not an output layer, so it cannot have Softmax activation");
            }
        }

        //Check passed: prepare for training
        intermediate_outputs.clear();
        initial_inputs.clear();
        optim->clear_state();

        layers.shrink_to_fit();

        enabled = true;
    }



    /**
     * Inserts `new_layer` at position `new_pos` in the network.
     * All layers at or after `new_pos` are moved one position backwards.
     * 
     * Inserting at position `{network}.layer_count()` will put the new layer at the end of the network.
     * 
     * Execution time scales linearly with the network's number of layers.
     * 
     * Network layers use 0-based indexing. The first layer is at index 0.
     * 
     * @param new_pos new position number to insert the layer. Must be between 0 and `{network}.layer_count()`, inclusive on both sides
     * @param new_layer layer to insert at position `new_pos`
     * @throws `illegal_state` if this method is called while enabled
     */
    void insert_layer_at(int new_pos, Layer new_layer) {
        assert((new_pos >= 0 && new_pos <= layer_count() && "New layer insertion position must be between 0 and (number of layers)"));
        if(enabled) {
            throw illegal_state("Cannot insert a new layer while enabled");
        }

        layers.insert(layers.begin() + new_pos, new_layer);
    }



    /**
     * Removes the layer whose name is `removal_name`.
     * 
     * The first layer in the network whose name matches (i.e. the layer with the lowest index)
     * will be removed.
     * 
     * If there are no matches, throws `std::out_of_range`.
     * 
     * @param layer_name layer name to remove
     * @throws `illegal_state` if the network is enabled
     * @throws `std::out_of_range` if no layer's name matches `removal_name`
     */
    void remove_layer(std::string removal_name) {
        if(enabled) {
            throw illegal_state("Cannot remove layers by name while the network is enabled");
        }

        for(unsigned int i=0; i<layers.size(); i++) {
            if(layers[i].name() == removal_name) {
                layers.erase(layers.begin() + i);
                return;
            }
        }
        throw std::out_of_range("No matches found");
    }



    /**
     * Removes the layer at position `remove_pos`.
     * 
     * @param remove_pos layer number to remove. Must be between 0 and `{network}.layer_count()`-1
     * @throws `illegal_state` if this method is called while the network is enabled
     */
    void remove_layer_at(int remove_pos) {
        assert((remove_pos>=0 && remove_pos<layer_count() && "Remove position must be between 0 and (number of layers)-1"));

        if(enabled) {
            throw illegal_state("Cannot remove a layer at a position if the network is enabled");
        }
        layers.erase(layers.begin() + remove_pos);
    }



    /**
     * Renames the layer at position `rename_pos` to `new_name`.
     * 
     * Unlike most setters, this method can be called, even if the network is enabled.
     * 
     * @param rename_pos layer number to rename (0-based indexing). Must be between 0 and `{network}.layer_count()`-1
     * @param new_name what to remane the layer at `rename_pos` to
     */
    void rename_layer_at(int rename_pos, std::string new_name) {
        assert((rename_pos>=0 && rename_pos<layer_count() && "Rename position must be between 0 and (number of layers)-1"));
        layers[rename_pos].set_name(new_name);
    }



    /**
     * Sets the activation function at layer `layer_number` to `new_activation_function`.
     * 
     * The input layer's number is 0. The first hidden layer's number is 1.
     * 
     * To remove a layer's activation function, set the layer's activation function to a `std::shared_ptr<IdentityActivation>`.
     * The IdentityActivation, f(x)=x, is a placeholder that does nothing.
     * 
     * @param layer_number which layer's biases to set (0-based indexing). Must be between 0 and {networkName}.layer_count()-1, inclusive on both ends
     * @param new_activation_function smart pointer to activation function to use 
     * @throws `illegal_state` if the network is enabled
     */
    void set_activation_function_at(int layer_number, std::shared_ptr<ActivationFunction> new_activation_function) {
        assert((layer_number>=0 && layer_number<(int)layers.size() && "To change activation functions, layer number must be between 0 and (number of layers)-1"));

        if(enabled) {
            throw illegal_state("Cannot change activation functions while the network is enabled");
        }

        layers[layer_number].set_activation_function(new_activation_function);
    }



    /**
     * Sets the bias vector at layer `layer_number` to `new_biases`.
     * 
     * The input layer's number is 0. The first hidden layer's number is 1.
     * 
     * @param layer_number which layer's biases to set (0-based indexing). Must be between 0 and {networkName}.layer_count()-1, inclusive on both ends
     * @param new_biases new bias vector to set. Must have {selected layer}.output_dimension() rows
     * @throws `illegal_state` if the network is enabled
     */
    void set_biases_at(int layer_number, Eigen::VectorXd new_biases) {
        assert((layer_number>=0 && layer_number<(int)layers.size() && "When changing bias vectors, layer number must be between 0 and (network's number of layers)-1"));
        assert((new_biases.rows() == layers[layer_number].output_dimension() && "New bias vector's number of rows must equal the selected layer's output dimension"));
        assert((new_biases.cols() == 1 && "New biases must be a column vector"));

        if(enabled) {
            throw illegal_state("Cannot manually set bias vectors while the network is enabled");
        }

        layers[layer_number].set_bias_vector(new_biases);
    }



    /**
     * Replaces the current loss calculator with `new_calculator`.
     * If no loss calculator is defined yet, the defined loss calculator becomes `new_calculator`.
     * 
     * If the network is enabled, this method throws `illegal_state`.
     * 
     * @param new_calculator smart pointer to loss calculator object
     * @throws `illegal_state` if the network is enabled
     */
    void set_loss_calculator(std::shared_ptr<LossCalculator> new_calculator) {
        if(enabled) {
            throw illegal_state("Network must not be enabled to update the loss calculator");
        }

        //Free any existing loss calculator
        if(loss_calc) {
            loss_calc.reset();
        }
        loss_calc = new_calculator;
    }



    /**
     * Replaces the curent optimizer with `new_optimizer`.
     * If no optimizer is defined yet, the defined optimizer becomes `new_optimizer`.
     * 
     * If the network is enabled, this method throws `illegal_state`.
     * 
     * @param new_optimizer smart pointer to desired optimizer object
     * @throws `illegal_state` if the network is enabled
     */
    void set_optimizer(std::shared_ptr<Optimizer> new_optimizer) {
        if(enabled) {
            throw illegal_state("Network must not be enabled to update the optimizer");
        }

        //Free any existing optimizer
        if(optim) {
            optim.reset();
        }
        optim = new_optimizer;
    }



    /**
     * Sets the optimizer's hyperparameters.
     * The purpose of each index in `hyperparameters` depends on the optimizer.
     * 
     * Example: For SGD optimizers, index 0 is the new learning rate, and index 1 is for the new momentum coefficient.
     * 
     * Throws `std::runtime_error` if no optimizer is defined.
     * 
     * @param hyperparameters std::vector of new hyperparameters to set (exact parameters and preconditions depends on the optimizer)
     * @throws `runtime_error` if no optimizer is defined
     */
    void set_optimizer_hyperparameters(const std::vector<double>& hyperparameters) {
        if(!optim) {
            throw illegal_state("The network needs to have a defined optimizer to set its hyperparameters");
        }
        optim->set_hyperparameters(hyperparameters);
    }



    /**
     * Sets the weights at layer `layer_number` to `new_weights`.
     * 
     * The input layer's number is 0. The first hidden layer's number is 1.
     * 
     * @param layer_number which layer's weights to set (0-based indexing). Must be between 0 and {networkName}.layer_count()-1, inclusive on both ends
     * @param new_weights new weight matrix for layer `layer_number`.
     * Number of rows must equal {selected layer}.output_dimension(), number of columns must equal {selected layer}.input_dimension()
     * @throws `illegal_state` if the network is enabled
     */
    void set_weights_at(int layer_number, Eigen::MatrixXd new_weights) {
        assert((layer_number>=0 && layer_number<(int)layers.size() && "When setting weight matrix, layer number must be between 0 and (network's number of layers)-1"));
        assert((new_weights.rows() == layers[layer_number].output_dimension() && "New weight matrix's row count must equal the layer's output dimension"));
        assert((new_weights.cols() == layers[layer_number].input_dimension() && "New weight matrix's column count must equal the layer's input dimension"));
    
        if(enabled) {
            throw illegal_state("Cannot manually set weight matrices when the network is enabled");
        }

        layers[layer_number].set_weight_matrix(new_weights);
    }



    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //METHODS (MAY BE CALLED ONLY IF THE NETWORK IS ENABLED)

public:
    /**
     * Returns the result of the feed-forward operation on the given input, i.e. the network's predictions for `input`.
     * 
     * If `training` is true, the network internally records layer outputs for backpropagation.
     * After using the method with `training`=true, the `reverse` method can be called.
     * All data from previous calls to `forward` are erased.
     * 
     * Requires that the network is enabled. If not, the method throws `illegal_state`.
     * 
     * @param input input to the network. Must have `{networkName}.input_dimension()` rows
     * @param training true if training the network, false if getting results for evaluation only. Default: `true`
     * @return the network's output, as a Eigen::VectorXd of dimension `{networkName}.output_dimension()`
     * @throws `illegal_state` if the network is not enabled
     */
    Eigen::VectorXd forward(const Eigen::VectorXd& input, bool training = true) {
        assert((input.cols() == 1 && "Input to forward operation must be a column vector"));
        assert((input.rows() == input_dimension() && "Input to forward operation must have same dimension as the network's input"));
        
        //Enable check
        if(!enabled) {
            throw illegal_state("Network forward and predict operation requires the network to be enabled");
        }

        if(training) {
            //set initial and intermediate outputs for 1 example

            initial_inputs.resize(1);
            initial_inputs[0] = input;

            intermediate_outputs.resize(1);
            intermediate_outputs[0] = {};
        }

        //Pass input through all layers' forward operations
        Eigen::VectorXd current_layer_output = input;
        for(int i = 0; i < (int)layers.size(); i++) {

            //Compute forward pass
            Eigen::VectorXd pre_activation = layers[i].forward(current_layer_output);
            current_layer_output = layers[i].activation_function()->compute(pre_activation);

            //Add intermediate outputs
            if(training) {
                intermediate_outputs[0].push_back({pre_activation, current_layer_output, layers[i].weight_matrix()});
            }
        }

        return current_layer_output;
    }


    /**
     * Returns the result of the feed-forward operation on all the given inputs,
     * i.e. the network's predictions for each element in `inputs`.
     * 
     * Operates with `n_threads` threads.
     * 
     * If `training` is true, the network internally records layer outputs for backpropagation.
     * After using the method with `training`=true, the `reverse` method can be called.
     * All data from previous calls to `forward` are erased.
     * 
     * Requires that the network is enabled. If not, the method throws `illegal_state`.
     * 
     * IMPORTANT: Double check the order of `n_threads` and `training`. Booleans are effectively integers.
     * 
     * @param inputs inputs to the network. Cannot be empty, and each element must have `{networkName}.input_dimension()` rows
     * @param n_threads number of threads to use. Must be on the interval [1, `inputs.size()`]. Default 1
     * @param training true if training the network, false if getting results for evaluation only. Default: `true`
     * @return the network's output, as a std::vector<Eigen::VectorXd>, whose elements are of dimension `{networkName}.output_dimension()`
     * @throws `illegal_state` if the network is not enabled
     */
    std::vector<Eigen::VectorXd> forward(const std::vector<Eigen::VectorXd> inputs, int n_threads = 1, bool training = true) {
        using namespace std;
        using namespace Eigen;

        assert(inputs.size() > 0 && "Inputs list cannot be empty");
        assert(n_threads > 0 && n_threads <= inputs.size() && "Number of threads in multi-input reverse operation must be between 1 and `inputs.size()`");
        
        //Enable check
        if(!enabled) {
            throw illegal_state("Network forward and predict operation requires the network to be enabled");
        }

        vector<VectorXd> outputs;
        outputs.resize(inputs.size());

        if(training) {
            //set initial inputs for the examples

            initial_inputs.resize(inputs.size());
            for(int i = 0; i < inputs.size(); i++) {
                assert(inputs[i].size() == input_dimension() && "All initial inputs in the multiple-input forward pass must have `input_dimension()` elements");
                initial_inputs[i] = inputs[i];
            }
            
            intermediate_outputs.clear();
            intermediate_outputs.resize(inputs.size());
        }
        
        #ifndef USING_EIGENLITE
            const int N_OLD_EIGEN_THREADS = Eigen::nbThreads();
            Eigen::setNbThreads(1);
        #endif

        std::vector<std::thread> threads(n_threads);
        std::atomic<size_t> next_thread_index{0};

        for (int t = 0; t < n_threads; ++t) {
            threads[t] = std::thread([&] {

                //loop through all inputs, deep-copying `next_thread_index` to each thread
                size_t current_input_index;
                while ((current_input_index = next_thread_index.fetch_add(1)) < inputs.size()) {

                    // Copy input
                    initial_inputs[current_input_index] = inputs[current_input_index];
                    intermediate_outputs[current_input_index].clear();

                    Eigen::VectorXd current_layer_output = inputs[current_input_index];

                    //Run the forward pass
                    for (size_t l = 0; l < layers.size(); l++) {
                        Eigen::VectorXd pre_activation_output = layers[l].forward(current_layer_output);
                        Eigen::VectorXd post_activation_output = layers[l].activation_function()->compute(pre_activation_output);

                        if (training) {
                            intermediate_outputs[current_input_index].push_back({
                                pre_activation_output, 
                                post_activation_output, 
                                layers[l].weight_matrix()
                            });
                        }

                        current_layer_output = std::move(post_activation_output);
                    }

                    //Move result into outputs
                    outputs[current_input_index] = std::move(current_layer_output);
                }
            });
        }

        for (auto& th : threads) {
            th.join();
        }
        
        #ifndef USING_EIGENLITE
            Eigen::setNbThreads(N_OLD_EIGEN_THREADS);
        #endif
        
        return outputs;
    }



    /**
     * Returns the network's predictions for `input`.
     * 
     * When this method is used, the network *does not* internally record intermediate layer outputs for backpropagation.
     * 
     * Equivalent to `{networkName}.forward(input, false)`.
     * 
     * Requires that the network is enabled.
     * 
     * @param input input to the network. Must have `{networkName}.input_dimension()` rows
     * @return the network's output, as a Eigen::VectorXd of dimension `{networkName}.output_dimension()`
     * @throws `illegal_state` if the network is not enabled
     */
    Eigen::VectorXd predict(const Eigen::VectorXd& input) {
        assert((input.cols() == 1 && "Input to network prediction must be a column vector"));
        assert((input.rows() == input_dimension() && "Input to prediction must have same dimension as the network's input"));

        return forward(input, false);
    }



    /**
     * Updates the weights and biases of this network using `predictions` and `actuals`, using the network's optimizer.
     * 
     * This method requires the network to be enabled. Also, `{networkName}.forward` with `training`=true must have been called since the network was enabled.
     * If these conditions are not met, the method throws `illegal_state`.
     * 
     * @param predictions what the network predicts for a given input
     * @param actuals expected output for the network's prediction
     * @throws `illegal_state` if the network is not enabled, or a feed-forward training operation was not done
     */
    void reverse(const Eigen::VectorXd& predictions, const Eigen::VectorXd& actuals) {
        assert((predictions.cols() == 1 && "Reverse process predictions must be a column vector"));
        assert((actuals.cols() == 1 && "Reverse process actuals must be a column vector"));
        assert((predictions.rows() == output_dimension() && "Reverse process predictions length must equal network's output dimension"));
        assert((actuals.rows() == output_dimension() && "Reverse process actuals length must equal network's output dimension"));
        assert(initial_inputs.size() == 1 && "Reverse process requires using `forward` on one input vector");
        assert(intermediate_outputs.size() == 1 && "Reverse process requires using `forward` on one input vector");

        //Enable check
        if(!enabled) {
            throw illegal_state("Reverse operation requires the network to be enabled");
        }
        //Ensure that there are intermediate outputs
        if(intermediate_outputs.size() <= 0) {
            throw illegal_state("Reverse operation requires that the `forward` method (with training = true) was previously used");
        }
        
        //Use the network's optimizer
        optim->step(layers, initial_inputs[0], intermediate_outputs[0], predictions, actuals, loss_calc);
    }


    /**
     * Updates the weights and biases of this network using `predictions` and `actuals`, using the network's optimizer.
     * 
     * This method requires the network to be enabled. Also, `{networkName}.forward` (for a std::vector of inputs, of size `predictions.size()`) 
     * with `training`=true must have been called since the network was enabled.
     * If these conditions are not met, the method throws `illegal_state`.
     * 
     * @param predictions what the network predicts for a given input. Cannot be empty, and all elements must be of dimension `{networkName}.output_dimension`
     * @param actuals expected output for the network's prediction. Must be of the same length as `predictions`, and all elements must be of dimension `{networkName}.output_dimension`
     * @param n_threads number of threads to use. Must be on the interval [1, `predictions.size()`]. Default 1.
     * @throws `illegal_state` if the network is not enabled, or a feed-forward training operation on many inputs was not done
     */
    void reverse(const std::vector<Eigen::VectorXd>& predictions, const std::vector<Eigen::VectorXd>& actuals, int n_threads = 1) {
        using namespace Eigen;

        assert(predictions.size() > 0 && "Predictions list for reverse batch training cannot be empty");
        assert(actuals.size() > 0 && "Actuals list for reverse batch training cannot be empty");
        assert(predictions.size() == actuals.size() && "Sizes of the predictions and actuals vectors must be equal");
        #ifndef NDEBUG //This is wacked.
            for(const VectorXd& p : predictions) {
                assert(p.size() == output_dimension() && "Multi-input reverse operation requires all predictions' dimensions to equal `output_dimension()`");
            }
            for(const VectorXd& a : actuals) {
                assert(a.size() == output_dimension() && "Multi-input reverse operation requires all actuals' dimensions to equal `output_dimension()`");
            }
        #endif 
        assert(n_threads > 0 && n_threads <= predictions.size() && "Number of threads in multi-input reverse operation must be between 1 and `predictions.size()`");
        
        //Enable check
        if(!enabled) {
            throw illegal_state("Reverse operation requires the network to be enabled");
        }
        
        //Ensure that there are enough intermediate outputs
        if(intermediate_outputs.size() != predictions.size()) {
            throw illegal_state("Reverse operation requires that the `forward` method (with training = true) was previously used with a batch of size `predictions.size()`");
        }

        optim->step_minibatch(layers, initial_inputs, intermediate_outputs, predictions, actuals, loss_calc, n_threads);
    }



    ////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////
    //OPERATOR OVERLOADS

    /**
     * Adds `new_layer` as the final layer of the network.
     * 
     * If the network is enabled, this method throws `illegal_state`.
     * 
     * Equivalent to `{networkName}.add_layer(new_layer)`.
     * 
     * @param new_layer layer to add
     * @throws `illegal_state` if the network is enabled
     */
    void operator+=(Layer new_layer) {
        if(enabled) {
            throw illegal_state("(+= operator) Cannot add layer while the network is enabled");
        }
        add_layer(new_layer);
    }



    /**
     * Exports `network` to the output stream `output_stream`, returning a reference to `output_stream` with `network` added.
     * 
     * The output stream will contain all layers converted to strings, separated by newlines.
     * 
     * @param output_stream output stream to export to
     * @param network network to export
     * @return new output stream containing the network's information inside
     */
    friend std::ostream& operator<<(std::ostream& output_stream, const Network& network);

    ////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////
    //DESTRUCTOR

    /**
     * Properly destroys a Network.
     */
    ~Network() {
        loss_calc.reset();
        optim.reset();
    }

};



std::ostream& operator<<(std::ostream& output_stream, const Network& network) {
    
    //enabled or disabled, layer count
    output_stream << "Network (" << (network.enabled ? "enabled" : "disabled") << "); ";
    output_stream << network.layer_count() << " layers, ";

    //loss calculator
    if(network.loss_calc) {
        output_stream << network.loss_calc->name() << " loss; ";
    }
    else {
        output_stream << "no defined loss; ";
    }

    //optimizer
    if(network.optim) {
        output_stream << network.optim->name() << " optimizer (";

        for(int h = 0; h < (int)network.optim->hyperparameters().size(); h++) {
            output_stream << network.optim->hyperparameters() [h];
            
            output_stream << ((h < (int)network.optim->hyperparameters().size() - 1) ? ", " : ")");
        }
        
    }
    else {
        output_stream << "no defined optimizer";
    }

    //layers
    if(network.layers.size() > 0) {
        output_stream << "\n";
        for(Layer l : network.layers) {
            output_stream << l << "\n";
        }
    }
    return output_stream;
}




}