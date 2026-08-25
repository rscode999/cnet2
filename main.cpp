#include <initializer_list>
#include <memory>
#include <xtensor/io/xio.hpp>

#include "include/network.hpp"

using namespace std;
using namespace xt;
using namespace cast;


int main() {
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
    /*
    Alternate architecture to check safeguards
    l1 2-4 > branch (1)  0 > l1 4-1                                   \\              > combiner (9)  
                         1 > branch (3)  1 > l1 4-1 (4) > sigmoid (6) > combiner (8) ^    
                                         2 > l1 4-1 (5) > sigmoid (7) 
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

    Linear1d l1(4, 3);
    wcout << net << endl;
}