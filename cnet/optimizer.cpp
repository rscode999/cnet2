#include "layer.cpp"
#include "loss_calculator.cpp"

#include <list>
#include <thread>
#include <vector>

#include <iostream>

#pragma once


namespace CNet {

/**
 * Gives the minimum of the two arguments
 */
#define min(a, b) ( (a) < (b) ? (a) : (b) )


/**
 * Abstract class for network optimizers.
 * 
 * The Optimizer's `step` method performs an optimization pass over a Network's layers.
 * `step` is a private method that can be called only from within a Network, and is inaccessible to users.
 * 
 * An Optimizer's hyperparameters can be adjusted with optimizer-specific setter methods,
 * or use the `set_hyperparameters` method.
 * 
 * Normally used in a Network when inside a smart pointer, specifically a `std::shared_ptr`.
 */
class Optimizer {

friend class Network;

public:

    /**
     * @return std::vector containing the optimizer hyperparameters, in the order required by `set_hyperparameters`
     */
    virtual std::vector<double> hyperparameters() const = 0;

    /**
     * @return the optimizer's identifying std::string.
     */
    virtual std::string name() const {
        return "optimizer";
    }

    /**
     * Sets the optimizer's hyperparameters.
     * The purpose of each index in `hyperparameters` depends on the optimizer.
     * 
     * Example: For SGD optimizers, index 0 is the new learning rate, index 1 is for the new momentum coefficient,
     * and index 2 is for the new batch size.
     * 
     * @param hyperparameters vector of new hyperparameters to set (exact parameters depends on the optimizer)
     */
    virtual void set_hyperparameters(const std::vector<double>& hyperparameters) = 0;

    /**
     * @return detailed information about the optimizer, including hyperparameters
     */
    virtual std::string to_string() const {
        return "optimizer";
    }

private:

    /**
     * Resets the optimizer to its pre-training state.
     * 
     * The reset allows the Optimizer to handle network architecture changes.
     * 
     * Called internally by a Network when the Network is disabled.
     */
    virtual void clear_state() = 0;

    /**
     * Updates `layers` in-place using the optimizer, using calculated gradients.
     * 
     * This is a private method. Not intended to be called by a user.
     * 
     * Mutates `layers` and this Optimizer object.
     * 
     * @param layers std::vector of layers to optimize
     * @param initial_input input value of the network
     * @param intermediate_outputs outputs of each layer before and after the layer's activation function is applied
     * @param predictions the output of the network for `initial_input`
     * @param actuals what the network should predict for `initial_input`
     * @param loss_calculator smart pointer to loss calculator object
     */
    virtual void step(std::vector<Layer>& layers, const Eigen::VectorXd& initial_input, const std::vector<LayerCache>& intermediate_outputs,
        const Eigen::VectorXd& predictions, const Eigen::VectorXd& actuals, const std::shared_ptr<LossCalculator> loss_calculator) = 0;
    
    /**
     * Updates `layers` in-place using the optimizer, using calculated gradients.
     * 
     * Initial inputs, intermediate layer outputs, predictions, and actuals are treated as a single minibatch.
     * Losses and momentums are averaged over all inputs.
     * 
     * Each training example is trained by a worker thread. During training, the number of threads reserved for Eigen becomes 1.
     * The Eigen thread count is reset once this method finishes.
     * 
     * This is a private method. Not intended to be called by a user.
     * 
     * Mutates `layers` and this Optimizer object.
     * 
     * @param layers std::vector of layers to optimize
     * @param initial_inputs input values of the network, as a std::vector
     * @param intermediate_outputs outputs of each layer, for each training example, before and after the layer's activation function is applied
     * @param predictions the output of the network for each corresponding value in `initial_inputs`
     * @param actuals what the network should predict for each corresponding value in `initial_inputs`
     * @param loss_calculator smart pointer to loss calculator object
     */ 
    virtual void step_minibatch(std::vector<Layer>& layers, const std::vector<Eigen::VectorXd>& initial_input, const std::vector<std::vector<LayerCache>>& intermediate_outputs,
        const std::vector<Eigen::VectorXd>& predictions, const std::vector<Eigen::VectorXd>& actuals, const std::shared_ptr<LossCalculator> loss_calculator, int n_threads) = 0;

public:
    /**
     * Properly destroys an Optimizer.
     */
    virtual ~Optimizer() = default;
};





/**
 * A Stochastic Gradient Descent (SGD) optimizer, capable of training in batches.
 *
 * The optimizer has an adjustable learning rate and momentum coefficient.
 *   
 * The optimizer updates weights and biases on every `batch_size`-th input.
 * No updates occur on inputs other than ever `batch_size`-th input.
 */
class SGD : public Optimizer {

friend class Network;


private:


    /**
     * Utility struct to store changes in weights and biases.
     * 
     * Changes are stored as two `std::vector`s: `dW` (weights) and `dB` (biases)
     */
    struct Gradients {
        /**
         * Changes in weight matrices
         */
        std::vector<Eigen::MatrixXd> dW;

        /**
         * Changes in bias vectors
         */
        std::vector<Eigen::VectorXd> dB;
    };



    /**
     * Learning rate, dictating the optimization step size. Must be positive.
     */
    double learn_rate;

    /**
     * Amount of momentum to use. Must be non-negative.
     */
    double momentum_coeff;

    /**
     * Holds per-layer biases accumulated over training.
     * 
     * Each index holds the total bias for the given layer.
     */
    std::list<Eigen::VectorXd> total_biases;

    /**
     * Holds per-layer weights accumulated over training.
     * 
     * Each index holds the total bias for the given layer.
     */
    std::list<Eigen::MatrixXd> total_weights;

    /**
     * Bias velocities for momentum SGD
     */
    std::vector<Eigen::VectorXd> velocity_biases;

    /**
     * Weight velocities for momentum SGD
     */
    std::vector<Eigen::MatrixXd> velocity_weights;


    /**
     * Returns the product of a softmax output's Jacobian matrix 
     * with a gradient vector.
     * 
     * Does not explicitly calculate the softmax Jacobian.
     * 
     * Used when a layer uses softmax activation, but cross-entropy loss is not used.
     * 
     * @param softmax_out the softmax layer's output
     * @param loss_grad the gradient of the loss with respect to the softmax output
     * @return Jacobian from `softmax_output` * `loss_grad`
     */
    Eigen::VectorXd softmax_jacobian_vector_product(const Eigen::VectorXd& softmax_out, const Eigen::VectorXd& loss_grad) const {
        double dot = softmax_out.cwiseProduct(loss_grad).sum(); //Takes dot product
        return softmax_out * (loss_grad - dot);
    }

public:


    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //CONSTRUCTOR

    /**
     * Creates a new SGD optimizer, loading it with the given hyperparameters `learning_rate`, `momentum_coefficient`, and `batch_size`.
     * 
     * @param learning_rate learning rate, for determining speed of convergence. Must be positive. Default 0.01
     * @param momentum_coefficient for determining amount of momentum to use. Cannot be negative. Default 0
     */
    SGD(double learning_rate = 0.01, double momentum_coefficient = 0) {
        assert((learning_rate>0 && "Learning rate must be positive"));
        assert((momentum_coefficient>=0 && "Momentum coefficient cannot be negative"));
        
        learn_rate = learning_rate;
        momentum_coeff = momentum_coefficient;

    }



    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //GETTERS

    /**
     * (DEPRECATED METHOD) Throws a `std::runtime_error`.
     */
    int batch_size() const {
        throw std::runtime_error("DEPRECATED");
    }

    /**
     * @return vector of 2 hyperparameters: learning rate (index 0), momentum coefficient (index 1)
     */
    std::vector<double> hyperparameters() const override {
        return {learn_rate, momentum_coeff};
    }


    /**
     * @return optimizer's learning rate
     */
    double learning_rate() const {
        return learn_rate;
    }


    /**
     * @return optimizer's momentum coefficient
     */
    double momentum_coefficient() const {
        return momentum_coeff;
    }

    
    /**
     * @return `"sgd"`, the optimizer's identifying string
     */
    std::string name() const override {
        return "sgd";
    }


    /**
     * @return string containing the optimizer's name, learning rate, and momentum coefficient
     */
    std::string to_string() const override {
        return "sgd, learning rate=" + std::to_string(learn_rate) + ", momentum coefficient=" + std::to_string(momentum_coeff);
    }


    ////////////////////////////////////////////////////////////////////////////////////
    //SETTERS


    /**
     * (DEPRECATED METHOD) Throws a `std::runtime_error`.
     */
    void set_batch_size(int new_batch_size) {
        throw std::runtime_error("DEPRECATED");
    }


    /**
     * Sets the SGD optimizer's hyperparameters to `hyperparameters`.
     * Index 0 contains the new learning rate. Index 1 contains the new momentum coefficient.
     * 
     * If the batch size is changed, the optimizer's training data will be reset.
     * 
     * @param hyperparameters vector of new hyperparameters. Must be of length 2, where index 0 is positive and index 1 is on the interval [0, 1]
     */
    void set_hyperparameters(const std::vector<double>& hyperparameters) override {
        assert((hyperparameters.size() == 2 && "SGD optimizer hyperparameter list must be of length 2"));
        assert((hyperparameters[0] > 0 && "SGD hyperparameter index 0 (new learning rate) must be positive"));
        assert((0 <= hyperparameters[1] && hyperparameters[1] <= 1 && "SGD hyperparameter index 0 (new learning rate) must be on the interval [0, 1]"));

        learn_rate = hyperparameters[0];
        momentum_coeff = hyperparameters[1];
    }


    /**
     * Sets the optimizer's learning rate to `new_learning_rate`.
     * @param new_learning_rate new rate. Must be positive.
     */
    void set_learning_rate(double new_learning_rate) {
        assert(new_learning_rate > 0 && "SGD new learning rate must be positive");
        learn_rate = new_learning_rate;
    }


    /**
     * Sets the optimizer's momentum coefficient to `new_momentum_coefficient`.
     * @param new_learning_rate new momentum coefficient. Must be on the interval [0, 1]
     */
    void set_momentum_coefficient(double new_momentum_coefficient) {
        assert(0 <= new_momentum_coefficient && new_momentum_coefficient <= 1 && "SGD new momentum coefficient must be on the interval [0, 1]");
        momentum_coeff = new_momentum_coefficient;
    }

private:

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //METHODS (PRIVATE)

    /**
     * Resets the optimizer to its pre-training state.
     * 
     * Resets biases, weights, momentum data, and number of samples trained.
     */
    void clear_state() override {
        total_biases.clear();
        total_weights.clear();
        velocity_biases.clear();
        velocity_weights.clear();
    }


    /**
     * Returns gradients in the weights and biases of `layers`, based on the given inputs.
     * 
     * @param layers std::vector of layers to optimize
     * @param initial_input value that was first given to the network
     * @param intermediate_outputs outputs of each layer before and after the layer's activation function is applied
     * @param predictions the output of the network for `initial_input`
     * @param actuals what the network should predict for `initial_input`
     * @param loss_calculator smart pointer to loss calculator object
     * @return gradient and bias changes
     */
    Gradients compute_gradients(
        const std::vector<Layer>& layers,
        const Eigen::VectorXd& initial_input, 
        const std::vector<LayerCache>& intermediate_outputs,
        const Eigen::VectorXd& predictions, 
        const Eigen::VectorXd& actuals,
        const std::shared_ptr<LossCalculator>& loss_calculator) const {
        
        assert((predictions.cols() == 1 && "Predicted values must be a column vector"));
        assert((predictions.rows() == layers.back().output_dimension() && "Predicted value vector must have dimension equal to the network's output dimension"));
        assert((actuals.cols() == 1 && "Actual values must be a column vector"));
        assert((actuals.rows() == layers.back().output_dimension() && "Actual value vector must have dimension equal to the network's output dimension"));

        Gradients output;
        output.dB.resize(layers.size());
        output.dW.resize(layers.size());

        Eigen::VectorXd delta = Eigen::VectorXd(1);
        auto final_activation = layers.back().activation_function();
        bool final_activation_using_softmax = final_activation->name() == "softmax";
        bool using_cross_entropy_loss = loss_calculator->name() == "cross_entropy";
        
        // Step 1: Compute dL/dy
        Eigen::VectorXd loss_grad = loss_calculator->compute_loss_gradient(predictions, actuals);
 
        // Step 2: Handle softmax Jacobian if needed
        bool activation_derivative_applied = false; //Ensures that, if softmax Jacobian is applied, it isn't applied again
        if (final_activation_using_softmax && !using_cross_entropy_loss) {
            // This is softmax + non-cross-entropy (e.g., MSE)
            delta = softmax_jacobian_vector_product(predictions, loss_grad);
            activation_derivative_applied = true;
        } 
        else {
            // For cross-entropy + softmax or any other case
            delta = loss_grad;
        }

        // Step 3: Apply activation derivative if applicable
        if (!(final_activation_using_softmax && using_cross_entropy_loss)
            && !activation_derivative_applied) { //Only do this step if the softmax Jacobian is not already applied
                
            if (final_activation->using_pre_activation()) {
                delta = delta.cwiseProduct(final_activation->compute_derivative(intermediate_outputs.back().pre_activation));
            } 
            else {
                delta = delta.cwiseProduct(final_activation->compute_derivative(intermediate_outputs.back().post_activation));
            }
        }


        //Compute for the other layers
        for(int l = static_cast<int>(layers.size()) - 1; l >= 0; l--) {

            //Get original post-activation of the previous layer
            Eigen::VectorXd previous_post_activation = (l > 0) 
                ? intermediate_outputs[l-1].post_activation
                : initial_input;


            // Weight gradient = delta * previous activation transposed
            Eigen::MatrixXd prev_post_activ_transpose = previous_post_activation.transpose();
            Eigen::MatrixXd dW_temp = delta * prev_post_activ_transpose;
             ^ this line returns a VectorXd when it should return a MatrixXd. Something with static dimensions or the transpose.
            output.dW[l] = dW_temp;
            // Bias gradient = delta
            output.dB[l] = delta;

            //propagate delta
            if (l > 0) {
                // Backpropagate delta to previous layer
                delta = intermediate_outputs[l].weights.transpose() * delta;

                // Apply hidden layer activation derivative
                const auto& previous_activation = layers[l-1].activation_function();

                const Eigen::VectorXd& previous_layer_output = previous_activation->using_pre_activation()
                    ? intermediate_outputs[l-1].pre_activation
                    : intermediate_outputs[l-1].post_activation;
                
                delta = delta.cwiseProduct(previous_activation->compute_derivative(previous_layer_output));
            }
        }

        return output;
    }


    /**
     * Updates `layers` in-place using SGD.
     * 
     * Mutates `layers` and this SGD object.
     * 
     * @param layers std::vector of layers to optimize
     * @param initial_input input value of the network
     * @param intermediate_outputs outputs of each layer before and after the layer's activation function is applied
     * @param predictions the output of the network for `initial_input`
     * @param actuals what the network should predict for `initial_input`
     * @param loss_calculator smart pointer to loss calculator object
     */
    void step(std::vector<Layer>& layers, const Eigen::VectorXd& initial_input, const std::vector<LayerCache>& intermediate_outputs,
        const Eigen::VectorXd& predictions, const Eigen::VectorXd& actuals, const std::shared_ptr<LossCalculator> loss_calculator) override {

        using namespace Eigen;

        Gradients grads = compute_gradients(layers, initial_input, intermediate_outputs, predictions, actuals, loss_calculator);

        //initialize velocities if not set
        if(velocity_weights.size() != layers.size() || velocity_biases.size() != layers.size()) {
            //clear states
            velocity_weights.clear();
            velocity_biases.clear();

            //reload weight matrices
            velocity_weights.resize(layers.size());
            for(size_t w = 0; w < layers.size(); w++) {
                velocity_weights[w] = MatrixXd::Zero(layers[w].output_dimension(), layers[w].input_dimension());
            }

            //reload bias vectors
            velocity_biases.resize(layers.size());
            for(size_t b = 0; b < layers.size(); b++) {
                velocity_biases[b] = VectorXd::Zero(layers[b].output_dimension());
            }
        }


        //update the layers
        for(size_t l = 0; l < grads.dW.size(); l++) {

            //get velocity
            velocity_weights[l] = momentum_coeff * velocity_weights[l] - learn_rate * grads.dW[l];
            velocity_biases[l] = momentum_coeff * velocity_biases[l] - learn_rate * grads.dB[l];

            //update weight matrix
            MatrixXd current_weight_matrix = layers[l].weight_matrix();
            current_weight_matrix += velocity_weights[l];
            layers[l].set_weight_matrix(current_weight_matrix);

            //update bias vector
            VectorXd current_bias_vector = layers[l].bias_vector();
            current_bias_vector += velocity_biases[l];
            layers[l].set_bias_vector(current_bias_vector);
        }
        
    }


    /**
     * Updates `layers` using SGD over a minibatch.
     * 
     * Mutates `layers` and this SGD object.
     * 
     * @param layers std::vector of layers to optimize
     * @param initial_inputs input values of the network, as a std::vector
     * @param intermediate_outputs outputs of each layer, for each training example, before and after the layer's activation function is applied
     * @param predictions the output of the network for each corresponding value in `initial_inputs`
     * @param actuals what the network should predict for each corresponding value in `initial_inputs`
     * @param loss_calculator smart pointer to loss calculator object
     */
    void step_minibatch(std::vector<Layer>& layers, const std::vector<Eigen::VectorXd>& initial_inputs, const std::vector<std::vector<LayerCache>>& intermediate_outputs,
        const std::vector<Eigen::VectorXd>& predictions, const std::vector<Eigen::VectorXd>& actuals, const std::shared_ptr<LossCalculator> loss_calculator, int n_threads) override {

        using namespace std;
        using namespace Eigen;
        
        #ifndef USING_EIGENLITE
            const int OLD_N_THREADS = Eigen::nbThreads();
            Eigen::setNbThreads(1);
        #endif

        vector<Gradients> gradients(predictions.size()); //Gradients across the minibatch. Size = batch size

        //Train the minibatch until all samples have been trained
        vector<thread> threads(n_threads);

        std::atomic<int> next_idx{0};

        for (int t = 0; t < n_threads; ++t) {
            threads[t] = std::thread([&] {
                int idx;
                while ((idx = next_idx.fetch_add(1)) < predictions.size()) {
                    gradients[idx] = compute_gradients(
                        cref(layers),
                        cref(initial_inputs[idx]),
                        cref(intermediate_outputs[idx]),
                        cref(predictions[idx]),
                        cref(actuals[idx]),
                        loss_calculator
                    );
                }
            });
        }
        for (auto& th : threads) {
            th.join();
        }

        //Initialize average gradients to 0
        Gradients average_gradients;
        average_gradients.dW.resize(layers.size());
        average_gradients.dB.resize(layers.size());
        for(size_t i = 0; i < layers.size(); i++) {
            average_gradients.dW[i] = MatrixXd::Zero(layers[i].output_dimension(), layers[i].input_dimension());
            average_gradients.dB[i] = VectorXd::Zero(layers[i].output_dimension());
        }
        
        //Average the calculated gradients
        for(const Gradients& g : gradients) {
            //Idiot check
            assert(g.dW.size() == layers.size());
            assert(g.dB.size() == layers.size());

            for(size_t i = 0; i < g.dW.size(); i++) {
                average_gradients.dW[i] += g.dW[i];
                average_gradients.dB[i] += g.dB[i];
            }   
        }
        for(size_t i = 0; i < average_gradients.dW.size(); i++) {
            average_gradients.dW[i] /= gradients.size();
            average_gradients.dB[i] /= gradients.size();
        }

        //initialize velocities if not set
        if(velocity_weights.size() != layers.size() || velocity_biases.size() != layers.size()) {
            //clear states
            velocity_weights.clear();
            velocity_biases.clear();

            //reload weight matrices
            velocity_weights.resize(layers.size());
            for(size_t w = 0; w < layers.size(); w++) {
                velocity_weights[w] = MatrixXd::Zero(layers[w].output_dimension(), layers[w].input_dimension());
            }

            //reload bias vectors
            velocity_biases.resize(layers.size());
            for(size_t b = 0; b < layers.size(); b++) {
                velocity_biases[b] = VectorXd::Zero(layers[b].output_dimension());
            }
        }

        //Update layers
        for(size_t l = 0; l < average_gradients.dW.size(); l++) {

            //get velocity
            velocity_weights[l] = momentum_coeff * velocity_weights[l] - learn_rate * average_gradients.dW[l];
            velocity_biases[l] = momentum_coeff * velocity_biases[l] - learn_rate * average_gradients.dB[l];

            //update layers
            layers[l].set_weight_matrix(layers[l].weight_matrix() + velocity_weights[l]);
            layers[l].set_bias_vector(layers[l].bias_vector() + velocity_biases[l]);
        }
        
        #ifndef USING_EIGENLITE
            Eigen::setNbThreads(OLD_N_THREADS);
        #endif
    }
};




/**
 * Returns a std::shared_ptr to the optimizer whose name is `name` (i.e. "sgd") and specified by the hyperparameters given in `hyperparameters`.
 * 
 * @param name name of desired optimizer
 * @param hyperparameters list of hyperparameters for the new optimizer. Must be accepted by the optimizer's `set_hyperparameters` method
 * @return optimizer with matching name
 * @throws `runtime_error` if no matching optimizer name is found
 */
std::shared_ptr<Optimizer> make_optimizer(const std::string& name, const std::vector<double>& hyperparameters) {
    using namespace std;

    if(name == "sgd") {
        assert(hyperparameters.size() == 2 && "SGD creation requires 3 hyperparameters");
        shared_ptr<SGD> out = make_shared<SGD>(hyperparameters[0], hyperparameters[1]);
        return out;
    }
    throw runtime_error("optimizer name not recognized");
}




}