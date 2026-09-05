#include "include/cunit.hpp"
#include "../include/cast.hpp"

#include <xtensor/containers/xarray.hpp>

using namespace cast;
using namespace cunit;
using namespace std;
using namespace xt;



/**
* Trains on the XOR dataset, checking that loss has decreased and that predictions are with 0.05 of their outputs.
*/
void test_train_xor() {
    Network net = Network();

    //(2,4) -> (4,1)
    net.add_operator(make_shared<Linear1d>(2, 4));
    net.add_operator(make_shared<Sigmoid>());
    net.add_operator(make_shared<Linear1d>(4, 1));

    shared_ptr<MeanSquaredError> loss_calc = make_shared<MeanSquaredError>();
    net.set_loss_calculator(loss_calc);
    net.set_optimizer(make_shared<SGD>(0.02, 0.9));
    
    net.enable();


    vector<xarray<double>> inputs = {
        xarray<double>{0, 0},
        xarray<double>{0, 1},
        xarray<double>{1, 0},
        xarray<double>{1, 1}
    };

    vector<xarray<double>> expected_outputs = {
        xarray<double>{0},
        xarray<double>{1},
        xarray<double>{1},
        xarray<double>{0}
    };


    //Train
    for(int e = 0; e < 1000; e++) {
        for(int i = 0; i < (int)inputs.size(); i++) {
            xt::xarray<double> prediction = net.forward(inputs[i]);
            net.backward(prediction, expected_outputs[i]);
            net.optimize();
        }
    }

    //Compute total loss one more time
    double final_loss = 0;
    for(int i = 0; i < (int)inputs.size(); i++) {
        xarray<double> prediction = net.forward(inputs[i]);
        final_loss += loss_calc->compute(prediction, expected_outputs[i]);
    }
    
    //Check that loss is below 1e-6
    CUnit::assert_true(final_loss < 1e-3, "Final calculated loss cannot exceed 1e-3 (got " + std::to_string(final_loss) + ")");

    //Check that the loss is below 1e-8
    double loss = 0;
    for(int i = 0; i < (int)inputs.size(); i++) {
        xarray<double> prediction = net.forward(inputs[i]);
        loss += loss_calc->compute(prediction, expected_outputs[i]);
        net.backward(prediction, expected_outputs[i]);
        net.optimize();
    }


  for(int i = 0; i < (int)inputs.size(); i++) {
        xarray<double> prediction = net.forward(inputs[i]);

        //Check that the prediction is of the proper shape
        CUnit::assert_true(prediction.shape() == expected_outputs[i].shape(), "Prediction and expected shapes not equal (expected outputs index " + std::to_string(i) + ")");
        //Prediction is within 0.05 of the expected output
        CUnit::assert_iterable_almost_equals(expected_outputs[i], prediction, 0.05);
  }
}




/**
* Builds a classifier involving multiple branches, trains it on the XOR dataset, and checks that the classifier converged.
*/
void test_branch_train_complex() {
    Network net;
    std::shared_ptr<MeanSquaredError> loss_calc = make_shared<MeanSquaredError>();
    net.set_loss_calculator(loss_calc);
    net.set_optimizer(make_shared<SGD>(0.02, 0.9));

    /*
    This is a very complicated XOR classifier.
    l1 2-4 > branch (1)  0 > l1 4-1                                                 > combiner (9)  
                         1 > branch (3)  1 > l1 4-1 (4) > sigmoid (6) > combiner (8) ^    
                                         2 > l1 4-1 (5) > sigmoid (7) ^
    */

    net.add_operator(make_shared<Linear1d>(2, 4));

    net.add_splitter(2);
    net.add_operator(make_shared<Linear1d>(4, 1));
    net.add_splitter(2, 1);
    net.add_operator(make_shared<Linear1d>(4, 1), 1);
    
    net.add_operator(make_shared<Linear1d>(4, 1), 2);
    net.add_operator(make_shared<Sigmoid>(), 1);
    net.add_operator(make_shared<Sigmoid>(), 2);
    net.add_combiner({2}, 1);

    net.add_combiner({1}, 0);
  
    net.enable();


    vector<xarray<double>> inputs = {
        xarray<double>{0, 0},
        xarray<double>{0, 1},
        xarray<double>{1, 0},
        xarray<double>{1, 1}
    };

    vector<xarray<double>> expected_outputs = {
        xarray<double>{0},
        xarray<double>{1},
        xarray<double>{1},
        xarray<double>{0}
    };

    
    for(int e = 0; e < 1000; e++) {
        for(int i = 0; i < (int)inputs.size(); i++) {
            xt::xarray<double> prediction = net.forward(inputs[i]);
            net.backward(prediction, expected_outputs[i]);
            net.optimize();
        }
    }


    for(int i = 0; i < (int)inputs.size(); i++) {
        xarray<double> prediction = net.forward(inputs[i]);

        //Check that the prediction is of the proper shape
        CUnit::assert_true(prediction.shape() == expected_outputs[i].shape(), "Prediction and expected shapes not equal (expected outputs index " + std::to_string(i) + ")");
        //Each prediction element is within 0.1 of the expected output
        CUnit::assert_iterable_almost_equals(expected_outputs[i], prediction, 0.05);
  }
}





int main() {
    test_train_xor();
    test_branch_train_complex();
}