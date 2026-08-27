#include <cstdint>
#include <initializer_list>
#include <memory>
#include <unordered_map>
#include <xtensor/io/xio.hpp>

#include "include/loss_calculator.hpp"
#include "include/network.hpp"

using namespace std;
using namespace xt;
using namespace cast;


int main() {
    Network net;

    net.add_operator(make_shared<Linear1d>(2, 4));
    net.add_operator(make_shared<Sigmoid>());
    net.add_operator(make_shared<Linear1d>(4, 1));

    shared_ptr<SGD> optim = make_shared<SGD>(0.05, 0.9);
    shared_ptr<MeanSquaredError> mse_loss = make_shared<MeanSquaredError>();
    net.set_optimizer(optim);
    net.set_loss_calculator(mse_loss);

    net.enable();
    wcout << net << endl;

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
      double loss = 0;
      for(int i = 0; i < (int)inputs.size(); i++) {
          xt::xarray<double> prediction = net.forward(inputs[i]);
          loss += mse_loss->compute(prediction, expected_outputs[i]);
          net.backward(prediction, expected_outputs[i]);
          net.optimize();
      }

      if(e % 100 == 0) {
          cout << "Loss: " << loss << std::endl;
      }
  }

  for(int i = 0; i < (int)inputs.size(); i++) {
      xt::xarray<double> prediction = net.forward(inputs[i]);
      cout << "Prediction for " << inputs[i] << ": " << prediction << endl;
  }
}