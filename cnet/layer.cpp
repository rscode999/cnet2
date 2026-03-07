#include "activation_function.cpp"

#include <memory>

#include <Eigen/Core>

#pragma once


namespace CNet {



    
/**
 * Utility struct for storing pre-activation results, post-activation results, and weight matrices
 * at each step of the forward process.
 * 
 * Used in backpropagation.
 */
struct LayerCache {
    /**
     * Outputs of a network's layer, before the layer's activation function is applied
     */
    Eigen::VectorXd pre_activation;

    /**
     * Outputs of a network's layer, after the layer's activation function is applied
     */
    Eigen::VectorXd post_activation;

    /**
     * Weight matrix of the layer. Allows for mutable copies of the matrix during optimization.
     */
    Eigen::MatrixXd weights;
};



/**
 * A linear layer in a network.
 * Can be an input, hidden layer, or output, depending on its position in the network.
 * 
 * Each layer has a weight matrix and separate bias vector.
 * They can be manually viewed or updated using getter and setter methods.
 */
class Layer {

private:
    
    /**
     * Holds the per-neuron weights for this layer- does not contain biases.
     */
    Eigen::MatrixXd weights;

    /**
     * Holds the biases. Must be a column vector (i.e. has 1 column).
     */
    Eigen::VectorXd biases;


    /**
     * Smart pointer to activation function object
     */
    std::shared_ptr<ActivationFunction> activation_fcn;

    
    /**
     * Identifier for this layer
     */
    std::string layer_name;


public:

    /**
     * Creates a new Layer and initializes all fields. The activation function is set to the identity function, f(x)=x.
     * 
     * All indices in the layer's weight matrix and bias vector are initialized to random values on the interval [-1, 1].
     * 
     * @param input_dimension number of inputs that the Layer takes in. Must be positive.
     * @param output_dimension number of outputs that the Layer gives. Must be positive.
     * @param name identifier for this Layer. Default: `"layer"`
     * @param initialization_scale_factor factor to multiply weights and biases by. Default 1.
     */
    Layer(int input_dimension, int output_dimension, std::string name = "layer", int initialization_scale_factor = 1) {
        assert((input_dimension>0 && "Input vector's dimension must be positive"));
        assert((output_dimension>0 && "Output vector's dimension must be positive"));

        //Initialize weights and biases to random values on [-1, 1]
        weights = Eigen::MatrixXd::Random(output_dimension, input_dimension);
        biases  = Eigen::VectorXd::Random(output_dimension, 1);
        //Scale them
        weights = weights * initialization_scale_factor;
        biases = biases * initialization_scale_factor;
        
        //set functions to default
        activation_fcn = std::make_shared<IdentityActivation>();

        //initialize name
        this->layer_name = name;
    }


    /**
     * Creates a new Layer and loads it with the provided fields.
     * 
     * The weights and biases are initialized to random numbers 
     * on the interval [-`initialization_scale_factor`, `initialization_scale_factor`].
     * 
     * @param input_dimension number of inputs that the Layer takes in. Must be positive.
     * @param output_dimension number of outputs that the Layer gives. Must be positive.
     * @param activation_function smart pointer to activation function object to use
     * @param name identifier for this Layer. Default: `"layer"`
     * @param initialization_scale_factor factor to multiply weights and biases by. Default 1.
     */
    Layer(int input_dimension, int output_dimension,
        std::shared_ptr<ActivationFunction> activation_function,
        std::string name = "layer",
        int initialization_scale_factor = 1) {

        assert((input_dimension>0 && "Input vector's dimension must be positive"));
        assert((output_dimension>0 && "Output vector's dimension must be positive"));

        //Initialize weights and biases to random values on [-1, 1]
        weights = Eigen::MatrixXd::Random(output_dimension, input_dimension);
        biases  = Eigen::VectorXd::Random(output_dimension);
        //Scale them
        weights = weights * initialization_scale_factor;
        biases = biases * initialization_scale_factor;
        
        //initialize activation function
        activation_fcn = activation_function;

        //initialize name
        this->layer_name = name;
    }


    ///////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////
    //GETTERS
    
    /**
     * @return (deep copy of) smart pointer to the layer's activation function
     */
    const std::shared_ptr<ActivationFunction> activation_function() const {
        return activation_fcn;
    }

    /**
     * @return the layer's bias vector, as an Eigen::VectorXd
     */
    Eigen::VectorXd bias_vector() const {
        return biases;
    }
    
    /**
     * @return the number of elements in the layer's input
     */
    int input_dimension() const {
        return weights.cols();
    }

    /**
     * @return the layer's name
     */
    std::string name() const {
        return layer_name;
    }

    /**
     * @return the number of elements in the layer's output
     */
    int output_dimension() const {
        return weights.rows();
    }


    /**
     * @return the layer's weight matrix, as an Eigen::MatrixXd
     */
    Eigen::MatrixXd weight_matrix() const {
        return weights;
    }


    ///////////////////////////////////////////////////////////////////////////////////
    //SETTERS

    /**
     * Sets the layer's activation function to `new_activation_function`, a smart pointer to an activation function object.
     * 
     * Note: Setting the activation function to a `std::shared_ptr<IdentityActivation>` effectively removes the layer's activation function.
     * 
     * @param new_activation_function smart pointer to new activation function
     */
    void set_activation_function(std::shared_ptr<ActivationFunction> new_activation_function) {  
        activation_fcn.reset(); //An activation function is guaranteed to exist, even after the layer is created
        activation_fcn = new_activation_function;
    }



    /**
     * Sets the layer's bias vector to `new_biases`.
     * 
     * @param new_biases new vector of biases for the layer to use. Must have `{layerName}.output_dimension()` rows
     */
    void set_bias_vector(Eigen::VectorXd new_biases) {
        assert((new_biases.rows() == output_dimension() && "New bias vector must be of the same dimension as the output"));
        biases = new_biases;
    }


    /**
     * Sets the layer's name to `new_name`.
     * @param new_name new name for the layer
     */
    void set_name(std::string new_name) {
        layer_name = new_name;
    }



    /**
     * Sets the layer's weight matrix to `new_weights`.
     * @param new_weights new matrix of weights for the layer to use. Must have `{layerName}.output_dimension()` rows and `{layerName}.input_dimension()` columns
     */
    void set_weight_matrix(Eigen::MatrixXd new_weights) {
        assert((new_weights.rows() == output_dimension() && "New weight matrix must have a row count equal to the output dimension"));
        assert((new_weights.cols() == input_dimension() && "New weight matrix must have a column count equal to the input dimension"));
        weights = new_weights;
    }




    ///////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////
    //METHODS

    
    /**
     * Returns the result of the layer's forward operation on `input`.
     * 
     * Multiplies `input` by the layer's weight matrix and adds the result to the bias vector.
     * Does not apply the activation function.
     * 
     * Transforms a `{layerName}.input_dimension()` column vector into a `{layerName}.output_dimension()` vector.
     * 
     * @param input vector to apply the forward operation to. Must have `{layerName}.input_dimension()` elements
     * @return vector, of length `{layerName}.output_dimension()`, after `input`'s forward process
     */
    Eigen::VectorXd forward(const Eigen::VectorXd& input) {
        assert((input.cols() == 1 && "The forward process's input must be a column vector"));
        assert((input.rows() == input_dimension() && "Forward process's input vector must have dimension equal to the weight matrix's number of columns"));

        return ((weights * input) + biases);
    }


    /**
     * Exports `layer` to the output stream `output_stream`, returning a reference to `output_stream` with `layer` added.
     * @param output_stream output stream to export to
     * @param layer layer to export
     * @return new output stream containing the layer's information inside
     */
    friend std::ostream& operator<<(std::ostream& output_stream, const Layer& layer);
};


std::ostream& operator<<(std::ostream& output_stream, const Layer& layer) {
    output_stream << "Layer \"" << layer.layer_name << "\" (" << layer.weights.cols() << ", " << layer.weights.rows() 
        << "), activation function: " << layer.activation_fcn->name();
    return output_stream;
}




}