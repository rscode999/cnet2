#include <cassert>
#include <cstdint>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <random>

/**
 * Indicates a dynamically selected matrix dimension
 */
#define Dynamic -100

/**
 * Indicates that Eigen Lite is being used (rather than the full Eigen distribution)
 */
#define USING_EIGENLITE

#pragma once


namespace Eigen {


/**
 * Used to distinguish calls to `VectorXd::Random(n_rows, range)` from calls to `MatrixXd::Random(n_rows, n_cols)`.
 */
struct VectorTag {};

struct MatrixTag {};


/**
 * Imitates Eigen's `Matrix` class
 */
template<typename T, int32_t static_rows, int32_t static_cols>
class Matrix {

private:

    /**
     * Stores the Matrix's data. Managed with `new`/`delete`,
     */
    T* contents;

    /**
     * Number of rows. Must be positive.
     */
    int32_t dynamic_rows;

    /**
     * Number of columns. Must be positive.
     */
    int32_t dynamic_cols;


    template<typename Tp, int32_t other_static_rows, int32_t other_static_cols>
    friend class Matrix;


    /**
     * Returns the index in the Matrix's contents required to access index `row_index` and `col_index`.
     * 
     * Automatically performs index access calculation.
     * 
     * @param row_index row number to access
     * @param col_index column number to access
     * @return index in `contents` at (`row_number`, `col_number`)
     */
    inline int32_t contents_index_at(int32_t row_index, int32_t col_index) const {
        return row_index * dynamic_cols + col_index;
    }


    /**
     * Private helper class to implement Eigen's comma initializer.
     */
    class CommaInitializer {
    private:
        Matrix& mat;
        int32_t index_;
        int32_t total_;

    public:
        CommaInitializer(Matrix& m, T value) : mat(m), index_(0), total_(mat.rows() * mat.cols()) {
            assert(mat.rows() * mat.cols() > 0);
            mat.contents[index_++] = value;
        }

        CommaInitializer& operator,(T value) {
            assert(index_ < mat.rows() * mat.cols() && "Comma initializer: Too many elements loaded");
            mat.contents[index_++] = value;
            return *this;
        }

        ~CommaInitializer() {
            // Check underflow when full expression ends
            assert(index_ == total_ && "Comma initializer: Too few elements loaded");
        }
    };




public:

    /**
     * Creates a 1x1 uninitialized matrix
     */
    Matrix() {
        static_assert(static_rows == Dynamic, "Default constructor is for Dynamic matrices only");
        static_assert(static_cols == Dynamic || static_cols == 1, "Default constructor is for Dynamic matrices or vectors only");

        contents = new T[1];
        dynamic_rows = 1;
        dynamic_cols = 1;
    }


    /**
     * Creates an uninitialized Matrix with `n_rows` rows and `n_cols` columns
     * @param n_rows Number of rows to use. Must be positive (and must equal the static row count, if not Dynamic)
     * @param n_cols Number of columns to use. Must be positive (and must equal the static row count, if not Dynamic)
     */
    Matrix(int32_t n_rows, int32_t n_cols) : dynamic_rows(n_rows), dynamic_cols(n_cols) {
        static_assert(static_rows > 0 || static_rows == Dynamic, "Static row count must be positive or equal Dynamic");
        static_assert(static_cols > 0 || static_cols == Dynamic, "Static column count must be positive or equal Dynamic");
        assert((n_rows == static_rows || static_rows == Dynamic) && "Matrix creation: Rows must be positive and equal static row count, or be dynamic");
        assert((n_cols == static_cols || static_cols == Dynamic) && "Matrix creation: Columns must be positive and equal static column count, or be dynamic");
    
        contents = new T[n_rows * n_cols];
    }


    /**
     * Creates an uninitialized vector (a Matrix with 1 column) with `n_rows` rows.
     * @param n_rows Number of rows to use. Must be positive (and must equal the static row count, if not Dynamic).
     */
    Matrix(int32_t n_rows) : dynamic_rows(n_rows) {
        static_assert(static_rows > 0 || static_rows == Dynamic, "Static row count must be positive or equal Dynamic");
        assert((n_rows == static_rows || static_rows == Dynamic) && "Matrix creation: Rows must be positive and static row count, or be dynamic");
        
        dynamic_cols = 1;
        contents = new T[n_rows];
    }
    
    

    /**
     * Copies `other` into a new matrix object.
     * 
     * The number of static rows and columns is retained.
     * 
     * @param other other matrix to copy
     */
    template<typename Tp, int32_t other_static_rows, int32_t other_static_cols>
    Matrix(const Matrix<Tp, other_static_rows, other_static_cols>& other) {

        this->dynamic_rows = other.rows();
        this->dynamic_cols = other.cols();

        contents = new T[other.rows() * other.cols()];
        std::copy(other.contents, other.contents + (other.rows() * other.cols()), contents);
    }


    /**
     * Copies `other` into a new matrix object with identical static row/column counts.
     * 
     * The number of static rows and columns is retained.
     * 
     * @param other other matrix to copy
     */
    Matrix(const Matrix& other) {

        this->dynamic_rows = other.rows();
        this->dynamic_cols = other.cols();

        contents = new T[other.rows() * other.cols()];
        std::copy(other.contents, other.contents + (other.rows() * other.cols()), contents);
    }


    /**
     * Copies the pointer to `other`'s memory into a new matrix object with identical static row/column counts.
     * @param other other matrix to copy
     */
    Matrix(Matrix&& other) noexcept {
        contents = other.contents;
        dynamic_rows = other.dynamic_rows;
        dynamic_cols = other.dynamic_cols;

        other.contents = nullptr;
        other.dynamic_rows = 0;
        other.dynamic_cols = 0;
    }



    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////

    /**
     * @return number of columns
     */
    int32_t cols() const {
        return dynamic_cols;
    }

    constexpr int32_t staticCols() {
        return static_cols;
    }

    /**
     * @return number of rows
     */
    int32_t rows() const {
        return dynamic_rows;
    }

    constexpr int32_t staticRows() {
        return static_rows;
    }

    /**
     * @return number of rows (for column vectors only)
     */
    int32_t size() const {
        static_assert(static_cols == 1, "Size operation is for column vectors only");
        return dynamic_rows;
    }



    /**
     * Returns a matrix containing `n_rows` rows and `n_cols` columns, all initialized to the value `constant_value`.
     * 
     * For Matrices with static row and column count equaling `Dynamic` only.
     * 
     * @param n_rows Number of rows in the new matrix. Must match the statically given row count
     * @param n_cols Number of columns in the new matrix. Must match the statically given column count
     * @param constant_value Value to initialize the matrix with
     */
    static Matrix Constant(int32_t n_rows, int32_t n_cols, const T& constant_value) {
        static_assert(static_rows == Dynamic, "Constant matrix creation is for row/column count equaling Dynamic only");
        static_assert(static_cols == Dynamic, "Constant matrix creation is for row/column count equaling Dynamic only (You may have used a Matrix operation on a Vector)");
        assert(n_rows > 0 && "Constant vector creation requires row count to be positive");
        assert(n_cols > 0 && "Constant vector creation requires column count to be positive");

        Matrix<T, Dynamic, Dynamic> result(n_rows, n_cols);

        for (int32_t r = 0; r < n_rows; r++) {
            for (int32_t c = 0; c < n_cols; c++) {
                result(r, c) = constant_value;
            }
        }
        
        return result;
    }

    /**
     * Returns a vector (Matrix with 1 static column) containing `n_rows`, all initialized to the value `constant_value`.
     * 
     * For vectors with static row count equaling `Dynamic` only.
     * 
     * @param n_rows Number of rows in the new vector. Must be positive
     * @param constant_value Value to initialize the vector with
     */
    static Matrix Constant(int32_t n_rows, const T& constant_value) {
        static_assert(static_rows == Dynamic, "Constant vector creation is for row count equaling Dynamic only");
        static_assert(static_cols == 1, "Constant-vector creation is for vectors only (i.e. static column count is 1)");
        assert(n_rows > 0 && "Constant vector creation requires row count to be positive");

        Matrix<T, static_rows, 1> result(n_rows, 1);

        for (int32_t r = 0; r < n_rows; r++) {
            result(r, 0) = constant_value;
        }
        
        return result;
    }



    /**
     * Returns the element-wise product of this matrix and another matrix.
     *
     * @param other The matrix to multiply. Must have the same number of rows and columns as this matrix.
     * @return New matrix containing the coefficient-wise product.
     */
    template<typename Tp, int32_t rhs_static_rows, int32_t rhs_static_cols> 
    Matrix<Tp, static_rows, static_cols> cwiseProduct(const Matrix<Tp, rhs_static_rows, rhs_static_cols>& other) const {
        assert(rows() == other.rows() && "cwiseProduct: Rows must match other matrix's row count");
        assert(cols() == other.cols() && "cwiseProduct: Columns must match other matrix's column count");

        Matrix result(rows(), cols());

        for (int32_t i = 0; i < rows() * cols(); i++) {
            result.contents[i] = contents[i] * other.contents[i];
        }

        return result;
    }


    /**
     * Returns the coefficient-wise exponential of the matrix.
     *
     * Applies the exponential function (e^x) to each coefficient independently.
     *
     * @return A new matrix where each element is exp(original_element).
     */
    Matrix exp() const {
        Matrix result(rows(), cols());
        for (int i = 0; i < rows() * cols(); ++i)
            result.contents[i] = std::exp(contents[i]);
        return result;
    }


    /**
     * Returns a matrix where the log is applied to each element
     * @return element-wise logarithm of this matrix
     */
    Matrix log() const {

        Matrix<T, static_rows, static_cols> output(dynamic_rows, dynamic_cols);
        for (int32_t r = 0; r < dynamic_rows; ++r) {
            for (int32_t c = 0; c < dynamic_cols; ++c) {
                output(r, c) = std::log((*this)(r, c));
            }
        }

        return output;
    }



    /**
     * Returns a copy of this matrix with all values less than or equal to `max_value`
     * 
     * @param max_value Maximum value allowed
     * @return matrix with no element greater than `max_value`
     */
    Matrix max(const T& max_value) const {
        Matrix<T, static_rows, static_cols> output(dynamic_rows, dynamic_cols);

        for (int32_t r = 0; r < rows(); r++) {
            for(int32_t c = 0; c < cols(); c++) {
                output(r, c) = ((*this)(r, c) < max_value) ? (*this)(r, c) : max_value;
            }
        }
        return output;
    }



    /**
     * Returns a copy of this matrix with all values greater than or equal to `min_value`
     * 
     * @param min_value Minimum value allowed
     * @return matrix with no element less than `min_value`
     */
    Matrix min(const T& min_value) const {
        Matrix<T, static_rows, static_cols> output(dynamic_rows, dynamic_cols);

        for (int32_t r = 0; r < rows(); r++) {
            for(int32_t c = 0; c < cols(); c++) {
                output(r, c) = ((*this)(r, c) > min_value) ? (*this)(r, c) : min_value;
            }
        }
        return output;
    }



    /**
     * Returns the maximum coefficient value.
     *
     * Iterates over all coefficients and returns the largest value.
     * The matrix must contain at least one element.
     *
     * @return The maximum coefficient stored in the matrix.
     */
    T maxCoeff() const {
        T max = contents[0];
        for (int i = 1; i < rows() * cols(); i++) {
            if (contents[i] > max) {
                max = contents[i];
            }
        }
        return max;
    }


    /**
     * Returns a matrix (of static dimensions `Dynamic` by `Dynamic`) of dimension `n_rows` by `n_cols` initialized to the range [-`range`, `range`].
     * 
     * For matrices of static dimensions `Dynamic` by `Dynamic` only.
     * 
     * @param n_rows Number of rows in the matrix. Must be positive.
     * @param n_cols Number of columns in the matrix. Must be positive.
     * @param range Maximum absolute value for each element in the matrix. Default 1.
     * @return matrix randomly initialized
     */
    static Matrix<T, Dynamic, Dynamic> Random(int32_t n_rows, int32_t n_cols, const T& range = 1) {
        static_assert(static_rows == Dynamic, "Random matrix creation is for row/column count equaling Dynamic only");
        static_assert(static_cols == Dynamic, "Random matrix creation is for row/column count equaling Dynamic only (If creating a vector, call with the VectorTag)");
        assert(n_rows > 0 && "Random vector creation requires row count to be positive");
        assert(n_cols > 0 && "Random vector creation requires column count to be positive");
        
        Matrix<T, Dynamic, Dynamic> result(n_rows, n_cols);

        // Random number generator
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dist(T(-1 * range), T(range));

        for (int32_t r = 0; r < n_rows; ++r) {
            for (int32_t c = 0; c < n_cols; ++c) {
                result(r, c) = dist(gen);
            }
        }

        return result;
    }

    /**
     * Returns a vector (i.e. static column count is 1) (of static dimensions `Dynamic` by 1) of dimension `n_rows` initialized to the range [-`range`, `range`]
     * 
     * When using the range argument, this method may be mistaken for the `MatrixXd::Random(n_rows, n_cols, range = 1)` call.
     * If so, call with the VectorTag: `VectorXd v = VectorXd::Random(n_rows, n_cols, VectorTag{})`.
     * 
     * @param n_rows Number of rows in the vector
     * @param range Maximum absolute value for each element in the vector. Default 1.
     * @param VectorTag Used to force the compiler to select the vector's method instead of the matrix method
     * @return vector randomly initialized
     */
    static Matrix<T, Dynamic, 1> Random(int32_t n_rows, const T& range = 1, VectorTag = {}) {
        static_assert(static_rows == Dynamic, "Random vector creation is for row/column count equaling Dynamic only");
        static_assert(static_cols == 1, "Random vector creation is for column vectors (i.e. static column count is 1) only");
        assert(n_rows > 0 && "Random vector creation requires row count to be positive");

        Matrix<T, Dynamic, 1> output(n_rows, 1);

        // Random number generator
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dist(T(-1 * range), T(range));

        for (int32_t r = 0; r < n_rows; ++r) {
            output(r, 0) = dist(gen);
        }

        return output;
    }


    /**
     * Returns the squared norm of all elements in the matrix
     * @return sum over all elements (i,j) of [matrix(i,j)^2]
     */
    T squaredNorm() const {
        T output = T();
        for (int32_t r = 0; r < dynamic_rows; ++r) {
            for (int32_t c = 0; c < dynamic_cols; ++c) {
                T val = (*this)(r, c);
                output += val * val;
            }
        }
        return output;
    }



    /**
     * Returns the sum of each element in the matrix.
     *
     * @return The sum of all matrix coefficients.
     */
    T sum() const {
        T output = T();
        for (int i = 0; i < rows() * cols(); ++i) {
            output += contents[i];
        }
        return output;
    }



    /**
     * @return the transpose of this matrix
     */
    Matrix transpose() const {
        Matrix<T, static_cols, static_rows> output(cols(), rows());

        for (int32_t c = 0; c < cols(); c++) {
            for (int32_t r = 0; r < rows(); r++) {
                output(c, r) = contents[contents_index_at(r, c)];
            }
        }
        
        return output;
    }



    /**
     * Returns a new matrix with `func` applied to each element
     * @param func function to apply
     * @return deep copy with `func` applied
     */
    template<typename UnaryFunc>
    Matrix unaryExpr(UnaryFunc func) const {
        Matrix<T, static_rows, static_cols> result(rows(), cols());

        for (int32_t r = 0; r < rows(); ++r) {
            for (int32_t c = 0; c < cols(); ++c) {
                result(r, c) = func((*this)(r, c));
            }
        }

        return result;
    }




    /**
     * Returns a matrix containing `n_rows` rows and `n_cols` columns, all initialized to the value 0.
     * 
     * For Matrices with static row and column count equaling `Dynamic` only. 
     * 
     * @param n_rows Number of rows in the new matrix. Must be positive
     * @param n_cols Number of columns in the new matrix. Must be positive
     */
    static Matrix Zero(int32_t n_rows, int32_t n_cols) {
        static_assert(static_rows == Dynamic, "Zero matrix creation is for row/column count equaling Dynamic only");
        static_assert(static_cols == Dynamic, "Zero matrix creation is for row/column count equaling Dynamic only");
        assert((n_rows > 0) && "Constant matrix creation: Rows must be positive");
        assert((n_cols > 0) && "Constant matrix creation: Columns must be positive");
        
        return Matrix::Constant(n_rows, n_cols, 0);
    }

    /**
     * Returns a vector (Matrix with 1 column) containing `n_rows` rows, all initialized to the value 0.
     * 
     * For vectors with static row count equaling `Dynamic` only.
     * 
     * @param n_rows Number of rows in the new matrix. Must be positive or equal `Dynamic`
     */
    static Matrix Zero(int32_t n_rows) {
        static_assert(static_rows == Dynamic, "Zero vector creation is for row count equaling Dynamic only");
        static_assert(static_cols == 1, "Zero-vector creation is for vectors only (i.e. static column count is 1)");
        assert((n_rows > 0) && "Constant vector creation: Rows must be positive");
        
        Matrix<T, Dynamic, 1> result(n_rows, 1);

        for (int32_t r = 0; r < n_rows; r++) {
            result(r, 0) = 0;
        }
        
        return result;
    }

    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////

    /**
     * Returns a reference to the coefficient at position (r, c).
     *
     * Provides mutable access to the matrix element at the given row and column.
     *
     * @param r Row index. Must satisfy 0 <= r < rows().
     * @param c Column index. Must satisfy 0 <= c < cols()
     * @return Reference to the element at position (r, c).
     */
    T& operator() (int r, int c) {
        assert(0 <= r && r < rows() && "Access: Row out of bounds");
        assert(0 <= c && c < cols() && "Access: Column out of bounds");
        return contents[contents_index_at(r,c)];
    }

    /**
     * Returns a constant reference to the coefficient at position (r, c).
     *
     * @param r Row index. Must satisfy 0 <= r < rows(). Default: none.
     * @param c Column index. Must satisfy 0 <= c < cols(). Default: none.
     * @return Const reference to the element at position (r, c).
     */
    const T& operator() (int r, int c) const {
        assert(0 <= r && r < rows() && "Access: Row out of bounds");
        assert(0 <= c && c < cols() && "Access: Column out of bounds");
        return contents[contents_index_at(r,c)];
    }


    /**
     * Returns a reference to the coefficient at position r.
     * 
     * For column vectors (i.e. static number of rows = 1) only.
     * 
     * @param r Row index. Must satisfy 0 <= r < rows().
     * @return reference to the element at position r.
     */
    T& operator() (int r) {
        static_assert(static_cols == 1, "Single-index access operator is for column vectors only");
        assert(0 <= r && r < rows() && "Access for vector: Row out of bounds");
        return contents[contents_index_at(r,0)];
    }

    /**
     * Returns a constant reference to the coefficient at position r.
     * 
     * For column vectors (i.e. static number of rows = 1) only.
     * 
     * @param r Row index. Must satisfy 0 <= r < rows().
     * @return constant reference to the element at position r.
     */
    const T& operator() (int r) const {
        static_assert(static_cols == 1, "Single-index access operator is for column vectors only");
        assert(0 <= r && r < rows() && "Access for vector: Row out of bounds");
        return contents[contents_index_at(r,0)];
    }


    /**
     * Assigns the contents of `other` to this matrix.
     * 
     * `other` must have the same static dimensions as this matrix.
     * There is strict comparison between positive static dimensions and `Dynamic`.
     * 
     * @param other other matrix to assign to
     * @return reference to this matrix after the assignment
     */
    Matrix& operator=(const Matrix<T, static_rows, static_cols>& other) {
        
        //No self-assignment check: Reallocation only occurs if dimensions are different.

        // Reallocate if dimensions differ
        if (dynamic_rows != other.dynamic_rows ||
            dynamic_cols != other.dynamic_cols) {

            delete[] contents;

            dynamic_rows = other.dynamic_rows;
            dynamic_cols = other.dynamic_cols;
            contents = new T[dynamic_rows * dynamic_cols];
        }
    

        // Copy elements
        for (int i = 0; i < dynamic_rows * dynamic_cols; ++i) {
            contents[i] = other.contents[i];
        }

        return *this;
    } 



    /**
     * Assigns the pointer to `other`'s contents to this matrix.
     * 
     * `other` must have the same static dimensions as this matrix.
     * There is strict comparison between positive static dimensions and `Dynamic`.
     * 
     * @param other other matrix to assign to
     * @return reference to this matrix after the assignment
     */
    Matrix& operator=(Matrix&& other) noexcept {
        if (this != &other) {
            delete[] contents;

            contents = other.contents;
            dynamic_rows = other.dynamic_rows;
            dynamic_cols = other.dynamic_cols;

            other.contents = nullptr;
            other.dynamic_rows = 0;
            other.dynamic_cols = 0;
        }
        return *this;
    }



    /**
     * Returns the result of adding a scalar to this matrix.
     *
     * Adds the scalar value to each coefficient independently.
     *
     * @param scalar The scalar value to add. Default: none.
     * @return A new matrix where each element equals original + scalar.
     */
    Matrix operator+(T scalar) const {
        Matrix result(rows(), cols());
        for (int i = 0; i < rows()*cols(); ++i) {
            result.contents[i] = contents[i] + scalar; 
        }
        return result;
    }



    /**
     * Returns the element-wise sum of this matrix and another matrix.
     *
     * @param other The matrix to add. Must have the same number of rows and columns as this matrix.
     * @return A new matrix containing the coefficient-wise sum.
     */
    Matrix operator+(const Matrix& other) const {
        assert(rows() == other.rows() && "Matrix + matrix operation requires matrices to have the same number of rows");
        assert(cols() == other.cols() && "Matrix + matrix operation requires matrices to have the same number of columns");

        Matrix result(rows(), cols());
        for (int i = 0; i < rows()*cols(); ++i) {
            result.contents[i] = contents[i] + other.contents[i];
        }
        return result;
    }



    /**
     * Adds `other` to this matrix.
     * @param other Matrix to add to this matrix. Must match this matrix's (dynamic) row and column counts.
     */
    template<typename Tp, int32_t other_static_rows, int32_t other_static_cols>
    void operator+=(const Matrix<Tp, other_static_rows, other_static_cols>& other) {
        assert(rows() == other.rows() && "+= operator: Row count must match");
        assert(cols() == other.cols() && "+= operator: Column count must match");

        for (int i = 0; i < rows()*cols(); ++i) {
            contents[i] += other.contents[i];
        }
    }



    /**
     * Returns the result of subtracting `scalar` from each element.
     *
     * Subtracts the scalar value from each coefficient independently.
     *
     * @param scalar The scalar value to subtract
     * @return A new matrix where each element is subtracted by `scalar`.
     */
    Matrix operator-(T scalar) const {
        Matrix result(rows(), cols());
        for (int i = 0; i < rows()*cols(); ++i) {
            result.contents[i] = contents[i] - scalar;
        }
        return result;
    }



    /**
     * Returns the element-wise difference between this matrix and `other`.
     *
     * @param other The matrix to subtract from this matrix. Must have the same number of rows and columns as this matrix.
     * @return A new matrix containing the coefficient-wise difference.
     */
    Matrix operator-(const Matrix& other) const {
        assert(rows() == other.rows() && "Matrix - matrix operation requires matrices to have the same number of rows");
        assert(cols() == other.cols() && "Matrix - matrix operation requires matrices to have the same number of columns");

        Matrix result(rows(), cols());
        for (int i = 0; i < rows()*cols(); ++i) {
            result.contents[i] = contents[i] - other.contents[i];
        }
        return result;
    }



    /**
     * Subtracts `other` from this matrix.
     * @param other Matrix to subtract from this matrix. Must match this matrix's (dynamic) row and column counts.
     */
    template<typename Tp, int32_t other_static_rows, int32_t other_static_cols>
    void operator-=(const Matrix<Tp, other_static_rows, other_static_cols>& other) {
        assert(rows() == other.rows() && "+= operator: Row count must match");
        assert(cols() == other.cols() && "+= operator: Column count must match");

        for (int i = 0; i < rows()*cols(); ++i) {
            contents[i] -= other.contents[i];
        }
    }


    /**
     * Returns a matrix with `scalar` multiplied by each element of `matrix`.
     * 
     * `scalar` appears on the right side of the expression.
     * 
     * @param scalar number to multiply
     * @return this matrix * `scalar`
     */
    Matrix operator*(const T& scalar) const {
        Matrix<T, static_rows, static_cols> result(rows(), cols());

        for (int32_t r = 0; r < rows(); ++r) {
            for (int32_t c = 0; c < cols(); ++c) {
                result(r, c) = scalar * (*this)(r, c);
            }
        }

        return result;
    }



    /**
     * Returns the result of dividing this matrix by a scalar.
     *
     * Divides each coefficient independently by the scalar value.
     * The scalar must not be zero.
     *
     * @param scalar The scalar divisor. Cannot be zero.
     * @return A new matrix where each element equals original / scalar.
     */
    Matrix operator/(T scalar) const {
        assert(scalar != 0 && "Matrix scalar division- cannot divide by 0");

        Matrix result(rows(), cols());
        for (int i = 0; i < rows()*cols(); i++) {
            result.contents[i] = contents[i] / scalar;
        }
        return result;
    }



    /**
     * Divides each element in this matrix by `scalar`.
     * @param scalar Value to divide each element by. Cannot be 0.
     */
    void operator/=(const T& scalar) {
        assert(scalar != 0 && "Divide-assign: Scalar to divide by cannot be 0");

        for (int i = 0; i < rows()*cols(); ++i) {
            contents[i] /= scalar;
        }
    }



    /**
     * Loads this matrix with the specified comma-separated elements.
     * 
     * The amount of elements loaded must equal `rows()`*`cols()`.
     */
    template<typename Tp>
    CommaInitializer operator<<(Tp value) {
        return CommaInitializer(*this, value);
    }



    /**
     * Exports `matr` to the output stream `output_stream`, returning `output_stream` with the matrix inside.
     * 
     * @param output_stream output stream to export the matrix to
     * @param matr matrix to export
     * @return `output_stream` with `matr` inside
     */
    template<typename CharT, typename Traits, typename Tp, int32_t m_static_rows, int32_t m_static_cols>
    friend std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& output_stream, const Matrix<Tp, m_static_rows, m_static_cols>& matr);

    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////

    /**
     * Properly destroys a matrix
     */
    ~Matrix() {
        delete[] contents;
        contents = nullptr;
    }
};




/**
 * Returns a matrix with `scalar` subtracted from each element of `matrix`.
 * 
 * `scalar` appears on the left side of the expression.
 * 
 * @param scalar number to subtract
 * @param matrix matrix to subtract from
 * @return `scalar` - `matrix`
 */
template<typename T, int32_t static_rows, int32_t static_cols>
Matrix<T, static_rows, static_cols> operator-(const T& scalar, const Matrix<T, static_rows, static_cols>& matrix)
{
    Matrix<T, static_rows, static_cols> result(matrix.rows(), matrix.cols());

    for (int32_t r = 0; r < matrix.rows(); ++r) {
        for (int32_t c = 0; c < matrix.cols(); ++c) {
            result(r, c) = scalar - matrix(r, c);
        }
    }

    return result;
}



/**
 * Returns a matrix with `scalar` multiplied by each element of `matrix`.
 * 
 * `scalar` appears on the left side of the expression.
 * 
 * @param scalar number to multiply
 * @param matrix matrix to multiply by
 * @return `scalar` * `matrix`
 */
template<typename T, int32_t static_rows, int32_t static_cols>
inline Matrix<T, static_rows, static_cols> operator*(const T& scalar, const Matrix<T, static_rows, static_cols>& matrix) {
    Matrix<T, static_rows, static_cols> result(matrix.rows(), matrix.cols());

    for (int32_t r = 0; r < matrix.rows(); ++r) {
        for (int32_t c = 0; c < matrix.cols(); ++c) {
            result(r, c) = scalar * matrix(r, c);
        }
    }

    return result;
}



/**
 * Returns the matrix product of `lhs` and `rhs`.
 *
 * Performs standard matrix multiplication.
 * The number of columns of this matrix must equal the number of rows of the other matrix.
 *
 * @param lhs Left-hand side of the multiplication
 * @param rhs Right-hand side in the multiplication. Must satisfy `lhs.cols()` == `rhs.rows()`.
 * @return A new matrix containing the matrix multiplication result.
 */
template<typename T, int32_t lhs_static_rows, int32_t lhs_static_cols, int32_t rhs_static_rows, int32_t rhs_static_cols>
inline Matrix<T, lhs_static_rows, rhs_static_cols> operator*(const Matrix<T, lhs_static_rows, lhs_static_cols>& lhs, const Matrix<T, rhs_static_rows, rhs_static_cols>& rhs) {
    assert(lhs.cols() == rhs.rows() && "Matrix multiplication requires left-hand side's columns to equal right side's number of rows");

    Matrix<T, lhs_static_rows, rhs_static_cols> result(lhs.rows(), rhs.cols());

    for (int i = 0; i < lhs.rows(); ++i) {
        for (int j = 0; j < rhs.cols(); ++j) {
            result(i,j) = T();
            for (int k = 0; k < lhs.cols(); ++k) {
                result(i,j) += lhs(i,k) * rhs(k,j);
            }
        }
    }
    return result;
}



template<typename T>
inline Matrix<T, Dynamic, Dynamic> operator*(const Matrix<T, Dynamic, 1>& lhs, const Matrix<T, 1, Dynamic>& rhs) {
    Matrix<T, Dynamic, Dynamic> result(lhs.rows(), rhs.cols());
    for (int i = 0; i < lhs.rows(); ++i)
        for (int j = 0; j < rhs.cols(); ++j)
            result(i,j) = lhs(i) * rhs(0,j);

    return result;
}




template<typename CharT, typename Traits, typename T, int32_t static_rows, int32_t static_cols>
std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& output_stream, const Matrix<T, static_rows, static_cols>& matr) {
    // Determine the width needed for each column
    int col_width = 0;

    for (int i = 0; i < matr.rows(); ++i) {
        for (int j = 0; j < matr.cols(); ++j) {
            // Round to 5 decimals for display
            double rounded = std::round(matr(i,j) * 1e5) / 1e5;

            // Measure the width of the number as string
            std::basic_ostringstream<CharT, Traits, std::allocator<CharT>> ss;
            ss << rounded;
            int len = static_cast<int>(ss.str().length());
            if (len > col_width) col_width = len;
        }
    }

    // Output the matrix row by row
    output_stream << "[";
    for (int i = 0; i < matr.rows(); ++i) {
        output_stream << ((i == 0) ? "[ " : " [ ");
        for (int j = 0; j < matr.cols(); ++j) {
            double rounded = std::round(matr(i,j) * 1e5) / 1e5;
            output_stream << std::setw(col_width) << rounded;
            if (j != matr.cols() - 1)
                output_stream << "  "; // extra space between columns
        }
        output_stream << ((i == matr.rows() - 1) ? " ]" : " ]\n");
    }
    output_stream << "]\n";

    return output_stream;
    
}




/**
 * A matrix, of type double, with dynamically allocated row/column counts.
 * 
 * Imitates Eigen's `MatrixXd`.
 */
typedef Matrix<double, Dynamic, Dynamic> MatrixXd;

/**
 * A vector, of type double, with dynamically allocated row count.
 * 
 * Imitates Eigen's `VectorXd`.
 */
typedef Matrix<double, Dynamic, 1> VectorXd;


}