#include <string>

#include <Eigen/Core>

#pragma once

namespace CNet {

    


/**
 * Abstract class for calculating loss
 */
class LossCalculator {
public:

    /**
     * Returns the loss (error) when measured between `predictions` and `actuals`.
     * 
     * `predictions` and `actuals` must be column vectors with the same length.
     * 
     * @param predictions model's predictions for a given input
     * @param actuals true values for model predictions
     * @return calculator's loss of the model predictions
     */
    virtual double compute_loss(const Eigen::VectorXd& predictions, const Eigen::VectorXd& actuals) = 0;

    /**
     * Returns the gradient of the losses when measured between `predictions` and `actuals`.
     * 
     * `predictions` and `actuals` must be column vectors with the same length.
     * 
     * @param predictions model's predictions for a given input
     * @param actuals true values for model predictions
     * @return calculator's loss gradient of the model predictions
     */
    virtual Eigen::VectorXd compute_loss_gradient(const Eigen::VectorXd& predictions, const Eigen::VectorXd& actuals) = 0;

    /**
     * @return the identifying string of the loss calculator
     * 
     * Typically, a name is the calculator's class name in all lowercase,
     * where each word is separated by an underscore.
     * Example: CrossEntropy -> `"cross_entropy"`
     */
    virtual std::string name() = 0;

    /**
     * Properly destroys a loss calculator
     */
    virtual ~LossCalculator() = default;
};



/**
 * Calculates cross-entropy loss.
 */
class CrossEntropy : public LossCalculator {
/*
When used with Softmax activation in backpropagation,
should be treated as a special case.
*/

public:

    /**
     * Creates a new Cross Entropy loss calculator
     */
    CrossEntropy() {
    }

    
    /**
     * Returns a Eigen::VectorXd containing the cross-entropy losses, 
     * when measured between `predictions` and `actuals`.
     * 
     * `predictions` and `actuals` must be column vectors with the same length.
     * 
     * @param predictions predictions for a given input
     * @param actuals true values for model predictions
     * @return cross entropy calculator's loss of the model predictions
     */
    double compute_loss(const Eigen::VectorXd& predictions, const Eigen::VectorXd& actuals) override {
        assert((predictions.cols() == 1 && "Predictions must be a column vector"));
        assert((actuals.cols() == 1 && "Actual network outputs must be a column vector"));
        assert((predictions.rows() == actuals.rows() && "Number of rows in predictions and actuals must be equal"));

        // To avoid log(0), add small epsilon
        constexpr double epsilon = 1e-12;

        // Clamp predictions to [epsilon, 1.0]
        Eigen::VectorXd clipped_preds = predictions.max(1.0).min(epsilon);

        // Cross-entropy loss: -sum(actual * log(predictions))
        double loss = -actuals.cwiseProduct(clipped_preds.log()).sum(); //-(actuals * clipped_preds.log()).sum();

        return loss;
    }

    
    /**
     * Returns the gradient of cross-entropy for `predictions` and `actuals`
     * 
     * Cross-entropy gradients equal the element-wise subtraction: `predictions` - `actuals`
     * 
     * @param predictions predictions for a given input
     * @param actuals true values for model predictions
     * @return gradient of cross-entropy calculator's loss of the model predictions
     */
    Eigen::VectorXd compute_loss_gradient(const Eigen::VectorXd& predictions, const Eigen::VectorXd& actuals) override {
        assert((predictions.cols() == 1 && "Predictions must be a column vector"));
        assert((actuals.cols() == 1 && "Actual network outputs must be a column vector"));
        assert((predictions.rows() == actuals.rows() && "Number of rows in predictions and actuals must be equal"));

        // Gradient = predictions - actuals
        return (predictions - actuals);  // Return Eigen::VectorXd
    }

    /**
     * @return `"cross_entropy"`, the identifying string for Cross Entropy loss calculators
     */
    std::string name() override {
        return "cross_entropy";
    }
};



/**
 * Calculates Mean Squared Error (MSE) loss
 */
class MeanSquaredError : public LossCalculator {

public:

    /**
     * Creates a new MSE loss calculator
     */
    MeanSquaredError() {
    }



    /**
     * Returns the mean-squared error (MSE) losses, 
     * when measured between `predictions` and `actuals`.
     * 
     * `predictions` and `actuals` must be column vectors with the same length.
     * 
     * @param predictions predictions for a given input
     * @param actuals true values for model predictions
     * @return MSE calculator's loss of the model predictions
     */
    double compute_loss(const Eigen::VectorXd& predictions, const Eigen::VectorXd& actuals) override {
        assert((predictions.cols() == 1 && "Predictions must be a column vector"));
        assert((actuals.cols() == 1 && "Actuals must be a column vector"));
        assert((predictions.rows() == actuals.rows() && "Predictions and actuals must have the same dimension"));

        // Compute the MSE
        Eigen::VectorXd output = actuals - predictions;
        return output.squaredNorm() / output.size();
    }


    /**
     * Returns the gradient of MSE for `predictions` and `actuals`
     * 
     * @param predictions predictions for a given input
     * @param actuals true values for model predictions
     * @return gradient of MSE calculator's loss of the model predictions
     */
    Eigen::VectorXd compute_loss_gradient(const Eigen::VectorXd& predictions, const Eigen::VectorXd& actuals) override {
        assert((predictions.cols() == 1 && "Predictions must be a column vector"));
        assert((actuals.cols() == 1 && "Actuals must be a column vector"));
        assert((predictions.rows() == actuals.rows() && "Predictions and actuals must have the same dimension for gradient calculation"));

        return (2.0 / predictions.rows()) * (predictions - actuals);
    }


    /**
     * @return `"mean_squared_error"`, the identifying string for a MSE loss calculator
     */
    std::string name() override {
        return "mean_squared_error";
    }
};




/**
 * Returns a std::shared_ptr to the loss calculator whose name is `name`.
 * 
 * @param name name of desired loss calculator
 * @return loss calculator with matching name
 * @throws `runtime_error` if no matching loss calculator name is found
 */
std::shared_ptr<LossCalculator> make_loss_calculator(const std::string& name) {
    using namespace std;

    if(name == "cross_entropy") {
        shared_ptr<CrossEntropy> out = make_shared<CrossEntropy>();
        return out;
    }
    else if(name == "mean_squared_error") {
        shared_ptr<MeanSquaredError> out = make_shared<MeanSquaredError>();
        return out;
    }
    throw runtime_error("unrecognized loss calculator name \"" + name + "\"");
}




}