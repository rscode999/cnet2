#include "cnet/core.cpp"

#include <iostream>

using namespace std;
using namespace Eigen;
using namespace CNet;


/**
 * Loads the model with precomputed weights and biases for a 2-input XOR classification problem,
 * then checks the forward process's output against the reference model's results.
 * 
 * Model architecture: 2d -> 3d -> 1d.
 */
void test_xor_2layer() {
    //The weights and biases came from a Pytorch model

    MatrixXd l0_weights(3, 2);
    l0_weights << -0.33610883, -0.07779725,
    -1.5301809, -1.5219239,
    -3.9380543, -3.8303926;

    MatrixXd l1_weights(1, 3);
    l1_weights << 0.16919717,  3.2087102,  -3.4358373;

    VectorXd l0_biases(3);
    l0_biases << -1.9276419,   1.7493086,   0.39942038;

    VectorXd l1_biases(1);
    l1_biases << -0.6983648;

    shared_ptr<Sigmoid> sigmoid = make_shared<Sigmoid>();

    Layer l0 = Layer(2, 3, sigmoid, "l0");
    Layer l1 = Layer(3, 1, "l1");
    l0.set_weight_matrix(l0_weights);
    l1.set_weight_matrix(l1_weights);
    l0.set_bias_vector(l0_biases);
    l1.set_bias_vector(l1_biases);

    shared_ptr<SGD> sgd = make_shared<SGD>();
    shared_ptr<MeanSquaredError> mse = make_shared<MeanSquaredError>();
    Network net;
    net.set_loss_calculator(mse);
    net.set_optimizer(sgd);
    net.add_layer(l0);
    net.add_layer(l1);
    
    net.enable();
    
    VectorXd in(2);
    in << 0,0;
    cout << "Results for 0,0:  " << net.forward(in) << endl;
    cout << "Expected for 0,0: -5.9605e-08\n" << endl;

    in << 0,1;
    cout << "Results for 0,1:  " << net.forward(in) << endl;
    cout << "Expected for 0,1: 1\n" << endl;

    in << 1,0;
    cout << "Results for 1,0:  " << net.forward(in) << endl;
    cout << "Expected for 1,0: 1\n" << endl;

    in << 1,1;
    cout << "Results for 1,1:  " << net.forward(in) << endl;
    cout << "Expected for 1,1: 3.5763e-07" << endl;

    sgd.reset();
}



/**
 * Loads the model with precomputed weights and biases for a 2-input XOR classification problem,
 * then checks the forward process's output against the reference model's results.
 * 
 * Architecture: 2d -> 1d.
 * 
 * Note that this model is poorly trained.
 */
void test_xor_1layer() {
    MatrixXd weights(1, 2);
    weights << 0.2617063,  0.27747557;

    VectorXd biases(1);
    biases << -0.19150648;

    shared_ptr<Sigmoid> sigmoid = make_shared<Sigmoid>();
    Layer layer = Layer(2, 1, sigmoid, "layer");
    layer.set_weight_matrix(weights);
    layer.set_bias_vector(biases);

    Network net = Network();
    shared_ptr<SGD> optimizer = make_shared<SGD>();
    shared_ptr<MeanSquaredError> loss_calc = make_shared<MeanSquaredError>();
    net.add_layer(layer);
    net.set_loss_calculator(loss_calc);
    net.set_optimizer(optimizer);

    net.enable();

    VectorXd in(2);

    in << 0,0;
    cout << "Results for 0,0:  " << net.forward(in, false) << endl;
    cout << "Expected for 0,0: 0.4523\n" << endl;

    in << 0,1;
    cout << "Results for 0,1:  " << net.forward(in, false) << endl;
    cout << "Expected for 0,1: 0.5215\n" << endl;

    in << 1,0;
    cout << "Results for 1,0:  " << net.forward(in, false) << endl;
    cout << "Expected for 1,0: 0.5175\n" << endl;

    in << 1,1;
    cout << "Results for 1,1:  " << net.forward(in, false) << endl;
    cout << "Expected for 1,1: 0.5861" << endl;

    optimizer.reset();
}



/**
 * Creates a new model, trains it on the XOR dataset, then evaluates its predictions
 */
void test_training_xor() {
    Network net = Network();

    shared_ptr<Sigmoid> sigmoid = make_shared<Sigmoid>();
    Layer layer0 = Layer(2, 4, sigmoid, "layer0");
    Layer layer1 = Layer(4, 1, "layer1");
    net.add_layer(layer0);
    net.add_layer(layer1);
    sigmoid.reset();

    shared_ptr<SGD> optimizer = make_shared<SGD>(0.02, 0.9); //Learning rate 0.02, momentum 0.9
    shared_ptr<MeanSquaredError> loss_calc = make_shared<MeanSquaredError>();
    
    net.set_loss_calculator(loss_calc);
    net.set_optimizer(optimizer);
    optimizer.reset();

    net.enable();

    //Add XOR inputs
    vector<VectorXd> inputs;
    VectorXd v(2);
    v << 0, 0;
    inputs.push_back(v);
    v << 0, 1;
    inputs.push_back(v);
    v << 1, 0;
    inputs.push_back(v);
    v << 1, 1;
    inputs.push_back(v);

    //Add corresponding outputs
    vector<VectorXd> correct_outputs;
    for(int i=1; i<=4; i++) {
        VectorXd out = VectorXd(1);
        if(i==2 || i==3) {
            out << 1;
        }
        else {
            out << 0;
        }
        correct_outputs.push_back(out);
    }

    //Train for 1000 epochs
    const int N_EPOCHS = 1000;
    for(int e=1; e<=N_EPOCHS; e++) {
        double current_loss = 0;

        for(int i=0; i<(int)inputs.size(); i++) {
            VectorXd output = net.forward(inputs[i]);
            current_loss += loss_calc->compute_loss(output, correct_outputs[i]);
            net.reverse(output, correct_outputs[i]);
        }

        if(e % 200 == 0) {
            cout << "Loss for " << e << " epochs: " << current_loss << endl;
        }
    }

    cout << N_EPOCHS << " epochs:\n" << endl;

    VectorXd in(2);
    in << 0,0;
    cout << "Results for 0,0:  " << net.forward(in, false) << endl;
    cout << "Expected for 0,0: 0\n" << endl;

    in << 0,1;
    cout << "Results for 0,1:  " << net.forward(in, false) << endl;
    cout << "Expected for 0,1: 1\n" << endl;

    in << 1,0;
    cout << "Results for 1,0:  " << net.forward(in, false) << endl;
    cout << "Expected for 1,0: 1\n" << endl;

    in << 1,1;
    cout << "Results for 1,1:  " << net.forward(in, false) << endl;
    cout << "Expected for 1,1: 0" << endl;
}



/**
 * Returns a VectorXd representing `decimal_number` as a binary number with `n_bits` bits.
 * 
 * The most significant digit is the highest index number of the output.
 * 
 * Helper to `test_training_binconvert` and `test_hot_swap`
 * 
 * @param n_bits number of bits in the binary number. Must be positive
 * @param decimal_number number to convert to binary. Must be on the interval [0, 2^`n_bits`-1]
 * @return `decimal_number` in binary
 */
VectorXd decimal_to_binary(int n_bits, int decimal_number) {
    assert((n_bits>0 && "Number of bits for dec->bin conversion must be positive"));
    assert((decimal_number>=0 && decimal_number<(int)pow(2, n_bits) && "Decimal number must be between 0 and 2^n_bits-1"));

    VectorXd output(n_bits);
    int current = decimal_number;
    for(int i=0; i<output.size(); i++) {
        output(i) = current % 2;
        current = current / 2;
    }

    return output;
}

/**
 * Returns a VectorXd of length `n_indices`. All indices are 0 except for index `input`, which is 1.
 * 
 * Helper to `test_training_binconvert` and `test_hot_swap`
 *  
 * @param n_indices number of indices in the output
 * @param input the index to make 1. Must be between 0 and `n_indices`-1
 * @return one-hot VectorXd of length `n_indices`
 */
VectorXd one_hot_vectorxd(int n_indices, int input) {
    assert((input>=0 && input<n_indices && "input for one-hot conversion must be between 0 and n_indices-1"));

    VectorXd output(n_indices);
    for(int i=0; i<output.size(); i++) {
        output(i) = (i==input) ? 1 : 0;
    }
    return output;
}

/**
 * Creates and trains a model to convert an input, in binary, to a one-hot output.
 * 
 * Uses ReLU and softmax activations, along with cross-entropy loss.
 */
void test_training_binconvert() {
    const int N_INPUTS = 5; //Arbitary positive constant
    const int N_OUTPUTS = round(pow(2, N_INPUTS)); //Equals 2^N_INPUTS
    const int N_EPOCHS = 2000; //Positive constant

    Network net = Network();

    shared_ptr<SGD> optimizer = make_shared<SGD>(0.005, 0.9);
    net.set_optimizer(optimizer);
    optimizer.reset();

    shared_ptr<CrossEntropy> loss_calc = make_shared<CrossEntropy>();
    // shared_ptr<MeanSquaredError> loss_calc = make_shared<MeanSquaredError>(); //Should also run, but with ~50% accuracy
    net.set_loss_calculator(loss_calc);
    //Keep the loss calculator outside the network for per-epoch loss calculations

    //add layers
    shared_ptr<Relu> relu_activ = make_shared<Relu>();
    shared_ptr<Softmax> softmax_activ = make_shared<Softmax>();
    net.add_layer(N_INPUTS, 20, relu_activ, "layer0");
    net.add_layer(20, 40, relu_activ, "layer1"); //test: change to softmax and ensure the "enable" check fails
    net.add_layer(40, N_OUTPUTS, softmax_activ, "layer2");
    relu_activ.reset();
    softmax_activ.reset();

    //get inputs and corresponding expected outputs
    vector<VectorXd> inputs;
    vector<VectorXd> expected_outputs;
    for(int i=0; i<N_OUTPUTS; i++) {
        VectorXd current_input = decimal_to_binary(N_INPUTS, i);
        inputs.push_back(current_input);

        VectorXd current_output = one_hot_vectorxd(N_OUTPUTS, i);
        expected_outputs.push_back(current_output);
    }

    //enable training
    net.enable();

    cout << "Training started for " << N_INPUTS << " inputs, " << N_EPOCHS << " epochs" << endl;

    //train
    for(int e=1; e<=N_EPOCHS; e++) {
        double current_loss = 0;
        for(int i=0; i<(int)inputs.size(); i++) {
            VectorXd current_result = net.forward(inputs[i]);
            current_loss += loss_calc->compute_loss(current_result, expected_outputs[i]);
            net.reverse(current_result, expected_outputs[i]);
        }

        if(e%200==0) {
            cout << "Total loss for " << e << " epochs: " << current_loss << endl;
        }
    }


    //test model on each of the inputs. Model output is determined to be the maximum value.
    int n_correct = 0;
    for(int i=0; i<N_OUTPUTS; i++) {
        VectorXd test_input = decimal_to_binary(N_INPUTS, i);
        VectorXd test_output = net.forward(test_input, false);

        //find maximum index
        double max = -99999;
        int max_index = -1;
        for(int v=0; v<test_output.size(); v++) {
            if(test_output(v) > max) {
                max = test_output(v);
                max_index = v;
            }
        }
        cout << "Max for " << i << ": " << max_index << endl;

        //update number of correct predictions
        if(max_index == i) {
            n_correct++;
        }
    }
    cout << "Number correct: " << n_correct << " out of " << N_OUTPUTS << endl;
}



/**
 * Adds and removes layers from a network
 */
void test_add_remove() {
    const int N_INPUTS = 32;
    const int N_OUTPUTS = 5;

    Network net = Network();
    cout << net << "\n" << endl;
    net.set_loss_calculator(make_shared<CrossEntropy>());
    net.set_optimizer(make_shared<SGD>());

    shared_ptr<Relu> relu = make_shared<Relu>();
    shared_ptr<Softmax> softmax = make_shared<Softmax>();

    net.add_layer(N_INPUTS, 64, relu, "input");
    net.add_layer(64, 32, relu, "hidden0");
    net.add_layer(32, 16, relu, "hidden1");
    net.add_layer(16, N_OUTPUTS, softmax, "output");
    cout << net << endl;

    //Add a layer
    net.add_layer(N_OUTPUTS, 1, "new_layer");
    cout << "After layer addition:\n";
    cout << net << endl;

    //Remove a layer
    net.remove_layer_at(0);
    net.remove_layer("hidden1");
    cout << "After layer removals:\n";
    cout << net << endl;

    //Insert new layers
    net.insert_layer_at(1, Layer(32, 15, "inserted_layer_0"));
    cout << "After layer insertion:\n";
    cout << net << endl;
    cout << "Inputs: " << net.input_dimension() << ", outputs: " << net.output_dimension() << "\n" << endl;

    //Attempt to enable the net with input/output incompatibility
    try {
        net.enable();
    }
    catch(illegal_state& e) {
        cout << "Enabling failed- ";
        cout << e.what() << "\n" << endl;
    }

    //Change layer
    net.remove_layer("inserted_layer_0");
    net.insert_layer_at(1, Layer(32, 16, "new_inserted_layer_0"));
    cout << "After layer correction:\n" << net << "\n" << endl;
    
    //Attempt to enable the net with softmax not in an output layer
    try {
        net.enable();
    }
    catch(illegal_state& e) {
        cout << "Enabling failed- ";
        cout << e.what() << "\n" << endl;
    }

    //Replace the softmax layer
    net.remove_layer_at(2);
    net.insert_layer_at(2, Layer(16, 5, relu, "softmax_layer_replacement"));
    cout << "After illegal softmax correction:\n" << net << "\n" << endl;

    net.enable();
    cout << net << endl;

    //Attempt to remove a layer while activated
    try {
        net.set_biases_at(0, VectorXd(net.layer_at(0).output_dimension()));
    }
    catch(illegal_state& e) {
        cout << "Layer bias vector change failed- ";
        cout << e.what() << "\n" << endl;
    }

    cout << "Add/remove test finished" << endl;
}



/**
 * Trains a network, changes its architecture, then resumes network training.
 * 
 * Uses the binary to one-hot conversion problem.
 */
void test_hot_swap() {

    const int N_INPUTS = 5; //Arbitary positive constant
    const int N_OUTPUTS = round(pow(2, N_INPUTS)); //Equals 2^N_INPUTS
    const int N_INITIAL_EPOCHS = 30; //Positive constant
    const int N_FINAL_EPOCHS = 500; //Positive constant, larger than `N_INITIAL_EPOCHS`

    Network net = Network();

    shared_ptr<SGD> optimizer = make_shared<SGD>(0.0005, 0.9);
    net.set_optimizer(optimizer);
    optimizer.reset();

    shared_ptr<MeanSquaredError> old_loss_calc = make_shared<MeanSquaredError>();
    net.set_loss_calculator(old_loss_calc);
    //Keep the loss calculator outside the network for per-epoch loss calculations

    //add layers
    shared_ptr<Relu> relu_activ = make_shared<Relu>();
    shared_ptr<Softmax> softmax_activ = make_shared<Softmax>();
    net.add_layer(N_INPUTS, 10, relu_activ, "original_input_layer");
    net.add_layer(10, N_OUTPUTS, relu_activ, "original_output_layer");

    //get inputs and corresponding expected outputs
    vector<VectorXd> inputs;
    vector<VectorXd> expected_outputs;
    for(int i=0; i<N_OUTPUTS; i++) {
        VectorXd current_input = decimal_to_binary(N_INPUTS, i);
        inputs.push_back(current_input);

        VectorXd current_output = one_hot_vectorxd(N_OUTPUTS, i);
        expected_outputs.push_back(current_output);
    }

    cout << "Old network:\n" << net << endl;

    //enable training
    net.enable();

    cout << "Training started for " << N_INPUTS << " inputs, " << N_INITIAL_EPOCHS << " epochs" << endl;

    //train
    for(int e=1; e<=N_INITIAL_EPOCHS; e++) {
        double current_loss = 0;
        for(int i=0; i<(int)inputs.size(); i++) {
            VectorXd current_result = net.forward(inputs[i]);
            current_loss += old_loss_calc->compute_loss(current_result, expected_outputs[i]);
            net.reverse(current_result, expected_outputs[i]);
        }

        if(e%5==0) {
            cout << "Total loss for " << e << " epochs: " << current_loss << endl;
        }
    }

    //test model on each of the inputs. Model output is determined to be the maximum value.
    int n_correct = 0;
    for(int i=0; i<N_OUTPUTS; i++) {
        VectorXd test_input = decimal_to_binary(N_INPUTS, i);
        VectorXd test_output = net.forward(test_input, false);

        //find maximum index
        double max = -99999;
        int max_index = -1;
        for(int v=0; v<test_output.size(); v++) {
            if(test_output(v) > max) {
                max = test_output(v);
                max_index = v;
            }
        }

        //update number of correct predictions
        if(max_index == i) {
            n_correct++;
        }
    }
    cout << "Number correct: " << n_correct << " out of " << N_OUTPUTS << endl;


    //Add layers and update the loss calculator
    net.disable();

    net.insert_layer_at(1, Layer(10, 20, relu_activ, "new_layer_0"));
    net.insert_layer_at(2, Layer(20, N_OUTPUTS, relu_activ, "new_layer_1"));
    net.remove_layer("original_output_layer");
    net.add_layer(N_OUTPUTS, N_OUTPUTS, softmax_activ, "new_output_layer");

    shared_ptr<CrossEntropy> new_loss_calc = make_shared<CrossEntropy>();
    net.set_loss_calculator(new_loss_calc);

    cout << "\nNew network:\n" << net << endl;
    net.enable();
    cout << "Training restarted for " << N_INPUTS << " inputs, " << N_FINAL_EPOCHS << " epochs" << endl;


    //Retrain
    for(int e=1; e<=N_FINAL_EPOCHS; e++) {
        double current_loss = 0;
        for(int i=0; i<(int)inputs.size(); i++) {
            VectorXd current_result = net.forward(inputs[i]);
            current_loss += new_loss_calc->compute_loss(current_result, expected_outputs[i]);
            net.reverse(current_result, expected_outputs[i]);
        }

        if(e%100==0) {
            cout << "Total loss for " << e << " epochs: " << current_loss << endl;
        }
    }

    //Re-evaluate
    n_correct = 0;
    for(int i=0; i<N_OUTPUTS; i++) {
        VectorXd test_input = decimal_to_binary(N_INPUTS, i);
        VectorXd test_output = net.predict(test_input);

        //find maximum index
        double max = -99999;
        int max_index = -1;
        for(int v=0; v<test_output.size(); v++) {
            if(test_output(v) > max) {
                max = test_output(v);
                max_index = v;
            }
        }

        //update number of correct predictions
        if(max_index == i) {
            n_correct++;
        }
    }
    cout << "Number correct: " << n_correct << " out of " << N_OUTPUTS << endl;
}



/**
 * Stores a network to a file, then loads the network from the file.
 */
void test_file_load() {
    Network net;

    //Load from an empty file
    ofstream input_file("test.txt");
    input_file << "";
    input_file.close();
    Network net2 = load_network_config("test.txt");
    cout << net2 << "\n" << endl;

    // Layers, loss calculator, optimizer
    shared_ptr<Relu> relu = make_shared<Relu>();
    net.add_layer(2, 3, relu, "my relu layer");
    net.add_layer(3, 3);
    net.set_loss_calculator(make_shared<MeanSquaredError>());
    net.set_optimizer(make_shared<SGD>(0.01, 0.9));
    store_network_config("test.txt", net);
    net2 = load_network_config("test.txt");
    cout << net2 << "\n" << endl;

    //Completely empty network
    net = Network();
    store_network_config("test.txt", net);
    net2 = load_network_config("test.txt");
    cout << net2 << "\n" << endl;

    //Layers, but no optimizer or loss calculator
    net = Network();
    net.add_layer(1, 1, " a ");
    net.add_layer(1, 3, "hello world");
    // cout << net << endl;
    store_network_config("test.txt", net);
    net2 = load_network_config("test.txt");
    cout << net2 << endl;
}



/**
 * Trains on the binary to one-hot dataset, but with multithreaded batch training
 */
void test_batch_training() {
    const int N_INPUTS = 5; //Arbitary positive constant
    const int N_OUTPUTS = round(pow(2, N_INPUTS)); //Equals 2^N_INPUTS
    const int N_EPOCHS = 2000; //Positive constant

    Network net = Network();

    shared_ptr<SGD> optimizer = make_shared<SGD>(0.005, 0.9);
    net.set_optimizer(optimizer);
    optimizer.reset();

    shared_ptr<CrossEntropy> loss_calc = make_shared<CrossEntropy>(); //Switch with MSE. Should get ~1/3 correct
    net.set_loss_calculator(loss_calc);
    //Keep the loss calculator outside the network for per-epoch loss calculations

    //add layers
    shared_ptr<Relu> relu_activ = make_shared<Relu>();
    shared_ptr<Softmax> softmax_activ = make_shared<Softmax>();
    net.add_layer(N_INPUTS, 20, relu_activ, "layer0");
    net.add_layer(20, 40, relu_activ, "layer1"); //test: change to softmax and ensure the "enable" check fails
    net.add_layer(40, N_OUTPUTS, softmax_activ, "layer2");
    relu_activ.reset();
    softmax_activ.reset();

    //get inputs and corresponding expected outputs
    vector<vector<VectorXd>> inputs;
    vector<vector<VectorXd>> expected_outputs;
    const int BATCH_SIZE = N_OUTPUTS / 4;
    assert(BATCH_SIZE > 0 && "Test invalid- batch size is zero");
    for(int i = 0; i < N_OUTPUTS / BATCH_SIZE; i++) {
        vector<VectorXd> current_input;
        vector<VectorXd> current_expected_output;
        for(int b = 0; b < BATCH_SIZE; b++) {
            current_input.push_back(decimal_to_binary(N_INPUTS, i * BATCH_SIZE + b));
            current_expected_output.push_back(one_hot_vectorxd(N_OUTPUTS, i * BATCH_SIZE + b));
        }
        
        inputs.push_back(current_input);
        expected_outputs.push_back(current_expected_output);
    }

    //enable training
    net.enable();

    cout << "Training started for " << N_INPUTS << " inputs, " << N_EPOCHS << " epochs" << endl;

    //train
    for(int e = 1; e <= N_EPOCHS; e++) {

        double current_loss = 0;
        for(int i = 0; i < N_OUTPUTS / BATCH_SIZE; i++) {
            vector<VectorXd> current_outputs = net.forward(inputs[i], BATCH_SIZE, true);
            net.reverse(current_outputs, expected_outputs[i], BATCH_SIZE);

            for(int l = 0 ; l < current_outputs.size(); l++) {
                current_loss += loss_calc->compute_loss(current_outputs[l], expected_outputs[i][l]);
            }
        }

        if(e%200==0) {
            cout << "Total loss for " << e << " epochs: " << current_loss << endl;
        }
    }


    //test model on each of the inputs. Model output is determined to be the maximum value.
    int n_correct = 0;
    for(int i=0; i<N_OUTPUTS; i++) {
        VectorXd test_input = decimal_to_binary(N_INPUTS, i);
        VectorXd test_output = net.forward(test_input, false);

        //find maximum index
        double max = -99999;
        int max_index = -1;
        for(int v=0; v<test_output.size(); v++) {
            if(test_output(v) > max) {
                max = test_output(v);
                max_index = v;
            }
        }
        cout << "Max for " << i << ": " << max_index << endl;

        //update number of correct predictions
        if(max_index == i) {
            n_correct++;
        }
    }
    cout << "Number correct: " << n_correct << " out of " << N_OUTPUTS << endl;
}


int main() {
    //Call testing functions as you wish.

    // test_xor_1layer();
    // test_xor_2layer();
    test_training_xor();
    // test_training_binconvert();
    // test_add_remove();
    // test_hot_swap();
    // test_file_load();
    // test_batch_training();
}
