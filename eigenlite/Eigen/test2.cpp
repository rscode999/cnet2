#include "matrix2.cpp"


/**
 * Tests constructors, Constant, Zero, Random
 */
void test_constructor_initializer() {
    using namespace std;
    using namespace EigenLite;

    //Static matrix
    Matrix<double, MATRIX> m1(2, 2);
    m1 << 1, 2, 3, 4;
    cout << m1 << endl; //[[1, 2], [3, 4]]

    //Constant
    Matrix<double, MATRIX> m5 = Matrix<double, MATRIX>::Constant(3, 3, -100);
    cout << m5 << endl; //3x3 Matrix filled the value -100
    VectorXd m6 = VectorXd::Constant(2, -69);
    cout << m6 << endl; //2-D vector with the value -69

    //Zero
    MatrixXd m7 = MatrixXd::Zero(2, 5);
    cout << m7 << endl; //2x5 zero matrix
    VectorXd m8 = VectorXd::Zero(1);
    cout << m8 << endl; //1-D zero vector

    //Random
    MatrixXd m9 = MatrixXd::Random(2, 2);
    cout << m9 << endl; //2x2 initialized to random numbers on [-1, 1]
    VectorXd m10 = VectorXd::Random(3, 1000, VectorTag{});
    cout << m10 << endl; //3D vector initialized to random numbers on [-1000, 1000]
}



/**
 * Tests copy constructor and `=` operator overload
 */
void test_ruleof3() {
    using namespace std;
    using namespace EigenLite;


    //COPY CONSTRUCTOR

    Matrix<double, MATRIX> m(2, 2);
    m << 1, 2, 3, 4;

    Matrix<double, MATRIX> other = m;

    other(0,0) = -1;
    cout << m; //[[1, 2], [3, 4]]
    cout << other; //[[-1, 2], [3, 4]]
    cout << endl;


    // = OPERATOR
    Matrix<double, MATRIX> other2 = Matrix<double, MATRIX>::Zero(2,3);
    other2 = other;
    other2(1,1) = -100;

    cout << m; //[[1, 2], [3, 4]]
    cout << other; //[[-1, 2], [3, 4]]
    cout << other2 << endl; //[[-1, 2], [3, -100]]


    //REASSIGNMENT
    MatrixXd def;
    def = MatrixXd::Constant(2, 2, 2);
    cout << def << endl; //[[2, 2], [2, 2]]
}



/**
 * Tests add, subtract, multiply (by scalar), divide
 */
void test_arithmetic() {
    using namespace std;
    using namespace EigenLite;

    //ADD
    MatrixXd m(2, 2);
    m << 1, 2, 3, 4;
    MatrixXd n(2, 2);
    n << 0, 1, 0, 0;

    //ADD (matrices)
    Matrix<double, MATRIX> result(2, 2);
    result = m + n;
    cout << result << endl; // [[1, 3], [3, 4]]

    //SUBTRACT (matrices)
    result = m - n;
    cout << result << endl; // [[1, 1], [3, 4]]

    //ADD (scalar)
    result = m + 1;
    cout << result << endl; // [[2, 3], [4, 5]]
    
    //SUBTRACT (scalar)
    result = m - 1;
    cout << result; // [[0, 1], [2, 3]]
    //SUBTRACT (scalar, on LHS)
    result = 1.0 - m;
    cout << result << endl; // [[0, -1], [-2, -3]]

    //MULTIPLY BY SCALAR
    result = m * 2;
    cout << result; // [[2, 4], [6, 8]]
    //MULTIPLY BY SCALAR (scalar on RHS)
    result = 3.0 * m;
    cout << result << endl; //[[3, 6], [9, 12]]

    //DIVIDE (scalar)
    result = m / 2;
    cout << result << endl; //[[0.5, 1], [1.5, 2]]
}



/**
 * Tests add-assign, subtract-assign, divide-assign
 */
void test_arithmetic_assigns() {
    using namespace std;
    using namespace EigenLite;

    MatrixXd m(2,2);
    m << 1, 2, 3, 4;
    MatrixXd n(2,2);
    n << 1, 0, 0, 0;

    //ADD ASSIGN
    m += n;
    cout << m << endl; //[[2, 2], [3, 4]]

    //SUBTRACT ASSIGN
    m -= n;
    cout << m << endl; //[[1, 2], [3, 4]]

    //DIVIDE ASSIGN (SCALAR)
    m /= 2;
    cout << m << endl; //[[0.5, 1], [1.5, 2]]
}



/**
 * Tests matrix multiplication and matrix-vector products
 */
void test_matrix_multiplication() {
    using namespace std;
    using namespace EigenLite;

    MatrixXd m(2, 3);
    m << 1, 2, 3, 4, 5, 6;

    MatrixXd n(3, 2);
    n << 1, 1, 2, 2, 1, 1;

    MatrixXd result = m * n;
    cout << m << endl;
    cout << n << endl;
    cout << result << endl; //[[8, 8], [20, 20]]

    result = n * m;
    cout << result << endl; //[[5, 7, 9], [10, 14, 18], [5, 7, 9]]

    VectorXd v(3);
    v << 5, 4, 1;
    result = m * v;
    cout << result << endl; // [[16], [46]]

    //3D Column Vector * 3D Row Vector = 3x3 Matrix
    m = MatrixXd(1, 3);
    m << 1, 2, 3;
    v << 1, 2, 3;
    result = v * m;
    cout << result << endl; //[[1, 2, 3], [2, 4, 6], [3, 6, 9]]
}



/**
 * Tests any methods that are not `transpose` or `unaryExpr`.
 */
void test_methods() {
    using namespace std;
    using namespace EigenLite; 

    MatrixXd m(2, 2);
    m << 1, 2, 3, 4;
    MatrixXd n(2, 2);
    n << 1, 1, 1, 0;

    //COEFF WISE PRODUCT
    cout << m.cwiseProduct(n) << endl; // [[1, 2], [3, 0]]

    //LOG (base e)
    m << 1, 1, 1, 2.718;
    cout << m.log() << endl; // [[0, 0], [0, ~1]]

    //EXP (e^A)
    m << 0, 0, 0, 1;
    cout << m.exp() << endl; // [[1, 1], [1, ~2.718]]

    //MAX (CLAMP)
    m << 1, 2, 3, 4;
    cout << m.max(3) << endl; // [[1, 2], [3, 3]]
    //MIN (CLAMP)
    cout << m.min(2) << endl; // [[2, 2], [3, 4]]

    //MAX COEFF
    cout << m.maxCoeff() << "\n" << endl; //4

    //SQUARED NORM
    cout << m.squaredNorm() << "\n" << endl; //30

    //SUM
    cout << m.sum() << "\n" << endl; //10
}




/**
 * Tests the transpose operation
 */
void test_transpose() {
    using namespace std;
    using namespace EigenLite; 

    MatrixXd m(3, 2);
    m << 1, 2, 3, 4, 5, 6;

    cout << m.transpose() << endl; //[[1, 3, 5], [2, 4, 6]]

    cout << m.transpose().transpose() << endl; //[[1, 2], [3, 4], [5, 6]]

    m = MatrixXd(3, 1);
    m << 1, 2, 3;
    cout << m.transpose() << endl; //Row vector [[1, 2, 3]]

    m = MatrixXd(1, 1);
    m << 1;
    cout << m.transpose() << endl; // [[1]]

    //This operation caused static type incompatibilities in the old Matrix
    m = MatrixXd(3, 1);
    m << 1, 2, 3;
    VectorXd v = VectorXd(3);
    v << 1, 2, 3;
    cout << m * v.transpose(); //[[1, 2, 3], [2, 4, 6], [3, 6, 9]]
    cout << ( (m * v.transpose()).staticType() == MATRIX ) << endl; //1 (true)
}




/**
 * Tests the unary expression operation
 */
void test_unary_expr() {
     using namespace std;
    using namespace EigenLite; 

    MatrixXd m(3, 2);
    m << 1, 1, 2, 2, 3, 3;

    MatrixXd result = m.unaryExpr(
        [] (double x) -> double {
            return x * x;
        }
    );

    cout << result << endl; //[[1, 1], [4, 4], [9, 9]]
}




void test_conversions() {
    using namespace std;
    using namespace EigenLite;

    VectorXd v(3);
    v << 1, 2, 3;
    MatrixXd m = v;
    cout << m << endl; //[[1], [2], [3]]

    // m = MatrixXd(3); //Should Static Assert
}



int main() {
    // test_constructor_initializer();
    // test_ruleof3();
    // test_arithmetic();
    // test_arithmetic_assigns();
    // test_matrix_multiplication();
    test_methods();
    // test_transpose();
    // test_unary_expr();
    // test_conversions();
}
