#include <cassert>
#include <cstdint>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <random>



/**
 * Indicates that Eigen Lite is being used (rather than the full Eigen distribution)
 */
#define USING_EIGENLITE

#pragma once


namespace Eigen {




/**
 * Defines the static type (i.e. matrix, column-vector) of a Matrix object
 */
enum MatrixStaticType {
    MATRIX,
    COLUMN_VECTOR
};


/**
 * Used to distinguish calls to `VectorXd::Random(n_rows, range)` from calls to `MatrixXd::Random(n_rows, n_cols)`.
 */
struct VectorTag {};

/**
 * Used to distinguish calls to `VectorXd::Random(n_rows, range)` from calls to `MatrixXd::Random(n_rows, n_cols)`.
 */
struct MatrixTag {};



/**
 * Imitates Eigen's `Matrix` class.
 * 
 * Has only one static argument, other than the datatype of the items stored.
 * 
 * The matrix's contents are always allocated to the heap.
 * 
 * @param T datatype of the Matrix
 * @param static_type `MatrixStaticType`
 */
template<typename T, MatrixStaticType static_type>
class Matrix {

private:

    /**
     * Stores the Matrix's data. Managed with `new`/`delete`,
     */
    T* contents;

    /**
     * Number of rows. Must be positive.
     */
    int32_t n_rows;

    /**
     * Number of columns. Must be positive.
     */
    int32_t n_cols;


    template<typename T_, MatrixStaticType static_type_>
    friend class Matrix; //Declares self as friend class


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
        return row_index * n_cols + col_index;
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
            assert(mat.rows() * mat.cols() > 0 && "Comma initialize requires matrix to be non-empty");
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
     * Creates an empty uninitialized matrix
     */
    Matrix() {
        contents = nullptr;
        n_rows = 0;
        n_cols = 0;
    }


    /**
     * Creates an uninitialized Matrix with `n_rows` rows and `n_cols` columns
     * @param n_rows Number of rows to use. Must be positive (and must equal the static row count, if not Dynamic)
     * @param n_cols Number of columns to use. Must be positive (and must equal the static row count, if not Dynamic)
     */
    Matrix(int32_t n_rows, int32_t n_cols) : n_rows(n_rows), n_cols(n_cols) {
        contents = new T[n_rows * n_cols];
    }


    /**
     * Creates an uninitialized vector (a Matrix with 1 column) with `n_rows` rows.
     * 
     * For matrices of static type 1 (i.e. column vectors) only.
     * 
     * @param n_rows Number of rows to use. Must be positive.
     */
    Matrix(int32_t n_rows) : n_rows(n_rows) {
        static_assert(static_type == COLUMN_VECTOR, "Row-only initialization is for column-vectors only");
        n_cols = 1;
        contents = new T[n_rows];
    }
    
    

    /**
     * Copies `other` into a new matrix object.
     * 
     * The number of static rows and columns is retained.
     * 
     * @param other other matrix to copy
     */
    template<typename Tp, MatrixStaticType other_static_type>
    Matrix(const Matrix<Tp, other_static_type>& other) {

        this->n_rows = other.rows();
        this->n_cols = other.cols();

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

        this->n_rows = other.rows();
        this->n_cols = other.cols();

        contents = new T[other.rows() * other.cols()];
        std::copy(other.contents, other.contents + (other.rows() * other.cols()), contents);
    }


    /**
     * Copies the pointer to `other`'s memory into a new matrix object with identical static row/column counts.
     * @param other other matrix to copy
     */
    Matrix(Matrix&& other) noexcept {
        contents = other.contents;
        n_rows = other.n_rows;
        n_cols = other.n_cols;

        other.contents = nullptr;
        other.n_rows = 0;
        other.n_cols = 0;
    }




    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////

    /**
     * @return number of columns
     */
    int32_t cols() const {
        return n_cols;
    }

    /**
     * @return number of rows
     */
    int32_t rows() const {
        return n_rows;
    }


    /**
     * @return Matrix's static type
     */
    constexpr MatrixStaticType staticType() {
        return static_type;
    }


    /**
     * @return number of rows (for column vectors only)
     */
    int32_t size() const {
        static_assert(static_type == COLUMN_VECTOR, "Size operation is for column vectors only");
        return n_rows;
    }



    /**
     * Returns a (statically defined) matrix containing `n_rows` rows and `n_cols` columns, all initialized to the value `constant_value`.
     * 
     * @param n_rows Number of rows in the new matrix. Must be positive
     * @param n_cols Number of columns in the new matrix. Must be positive
     * @param constant_value Value to initialize the matrix with
     */
    static Matrix Constant(int32_t n_rows, int32_t n_cols, const T& constant_value) {
        assert(n_rows > 0 && "Constant matrix creation requires row count to be positive");
        assert(n_cols > 0 && "Constant matrix creation requires column count to be positive");

        Matrix<T, MATRIX> result(n_rows, n_cols);

        for (int32_t r = 0; r < n_rows; r++) {
            for (int32_t c = 0; c < n_cols; c++) {
                result(r, c) = constant_value;
            }
        }
        
        return result;
    }

    /**
     * Returns a (statically typed) vector containing `n_rows`, all initialized to the value `constant_value`.
     * 
     * For vectors with static row count equaling `Dynamic` only.
     * 
     * @param n_rows Number of rows in the new vector. Must be positive
     * @param constant_value Value to initialize the vector with
     */
    static Matrix Constant(int32_t n_rows, const T& constant_value) {
        static_assert(static_type == COLUMN_VECTOR, "Constant vector creation is for statically typed vectors");
        assert(n_rows > 0 && "Constant vector creation requires row count to be positive");

        Matrix<T, COLUMN_VECTOR> result(n_rows);

        for (int32_t r = 0; r < n_rows; r++) {
            result(r, 0) = constant_value;
        }
        
        return result;
    }



    /**
     * Returns the element-wise product of this matrix and another matrix.
     * 
     * The output's static type matches that of this object.
     *
     * @param other The matrix to multiply. Must have the same number of rows and columns as this matrix.
     * @return New matrix containing the coefficient-wise product.
     */
    template<typename Tp, MatrixStaticType other_static_type> 
    Matrix<Tp, static_type> cwiseProduct(const Matrix<Tp, other_static_type>& other) const {
        static_assert(other_static_type == static_type, "cWiseProduct: Static types of operands must match");
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

        Matrix output(n_rows, n_cols);
        for (int32_t r = 0; r < n_rows; ++r) {
            for (int32_t c = 0; c < n_cols; ++c) {
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
        Matrix output(n_rows, n_cols);

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
        Matrix output(n_rows, n_cols);

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
     * Returns a matrix of dimension `n_rows` by `n_cols` initialized to the range [-`range`, `range`].
     * 
     * @param n_rows Number of rows in the matrix. Must be positive.
     * @param n_cols Number of columns in the matrix. Must be positive.
     * @param range Maximum absolute value for each element in the matrix. Default 1.
     * @return matrix randomly initialized
     */
    static Matrix Random(int32_t n_rows, int32_t n_cols, const T& range = 1, MatrixTag = {}) {
        assert(n_rows > 0 && "Random vector creation requires row count to be positive");
        assert(n_cols > 0 && "Random vector creation requires column count to be positive");
        
        Matrix result(n_rows, n_cols);

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
     * Returns a vector of dimension `n_rows` initialized to the range [-`range`, `range`]
     * 
     * When using the range argument, this method may be mistaken for the `MatrixXd::Random(n_rows, n_cols, range = 1)` call.
     * If so, call with the VectorTag: `VectorXd v = VectorXd::Random(n_rows, n_cols, VectorTag{})`.
     * 
     * @param n_rows Number of rows in the vector
     * @param range Maximum absolute value for each element in the vector. Default 1.
     * @param VectorTag Used to force the compiler to select the vector's method instead of the matrix method
     * @return vector randomly initialized
     */
    static Matrix<T, COLUMN_VECTOR> Random(int32_t n_rows, const T& range = 1, VectorTag = {}) {
        static_assert(static_type == COLUMN_VECTOR, "Random vector creation is for statically typed column-vectors only");
        assert(n_rows > 0 && "Random vector creation requires row count to be positive");

        Matrix<T, COLUMN_VECTOR> output(n_rows);

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
        for (int32_t r = 0; r < n_rows; ++r) {
            for (int32_t c = 0; c < n_cols; ++c) {
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
     * 
     * Always returns a statically-typed matrix (not a column vector).
     */
    Matrix<T, MATRIX> transpose() const {
        Matrix<T, MATRIX> output(cols(), rows());

        for (int r = 0; r < rows(); ++r)
            for (int c = 0; c < cols(); ++c)
                output(c, r) = (*this)(r, c);

        return output;
    }



    /**
     * Returns a new matrix with `func` applied to each element
     * @param func function to apply
     * @return deep copy with `func` applied
     */
    template<typename UnaryFunc>
    Matrix unaryExpr(UnaryFunc func) const {
        Matrix result(rows(), cols());

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
     * @param n_rows Number of rows in the new matrix. Must be positive
     * @param n_cols Number of columns in the new matrix. Must be positive
     */
    static Matrix Zero(int32_t n_rows, int32_t n_cols) {
        assert((n_rows > 0) && "Constant matrix creation: Rows must be positive");
        assert((n_cols > 0) && "Constant matrix creation: Columns must be positive");
        
        return Matrix::Constant(n_rows, n_cols, 0);
    }

    /**
     * Returns a statically-typed column vector containing `n_rows` rows, all initialized to the value 0.
     * 
     * For vectors with static row count equaling `Dynamic` only.
     * 
     * @param n_rows Number of rows in the new matrix. Must be positive or equal `Dynamic`
     */
    static Matrix<T, COLUMN_VECTOR> Zero(int32_t n_rows) {
        static_assert(static_type == COLUMN_VECTOR, "Zero-vector creation is for vectors only (i.e. static column count is 1)");
        assert((n_rows > 0) && "Constant vector creation: Rows must be positive");
        
        Matrix<T, COLUMN_VECTOR> result(n_rows);

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
        static_assert(static_type == COLUMN_VECTOR, "Single-index access operator is for column vectors only");
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
        static_assert(static_type == COLUMN_VECTOR, "Single-index access operator is for column vectors only");
        assert(0 <= r && r < rows() && "Access for vector: Row out of bounds");
        return contents[contents_index_at(r,0)];
    }


    /**
     * Assigns the contents of `other` to this matrix.
     * 
     * `other` must have the same static type as this matrix.
     * 
     * @param other other matrix to assign to
     * @return reference to this matrix after the assignment
     */
    Matrix& operator=(const Matrix<T, static_type>& other) {
        
        //No self-assignment check: Reallocation only occurs if dimensions are different.

        // Reallocate if dimensions differ
        if (n_rows != other.n_rows ||
            n_cols != other.n_cols) {

            delete[] contents;

            n_rows = other.n_rows;
            n_cols = other.n_cols;
            contents = new T[n_rows * n_cols];
        }
    

        // Copy elements
        for (int i = 0; i < n_rows * n_cols; ++i) {
            contents[i] = other.contents[i];
        }

        return *this;
    } 



    /**
     * Assigns the pointer to `other`'s contents to this matrix.
     * 
     * `other` must have the same static type as this matrix.
     * 
     * @param other other matrix to assign to
     * @return reference to this matrix after the assignment
     */
    Matrix& operator=(Matrix&& other) noexcept {
        if (this != &other) {
            delete[] contents;

            contents = other.contents;
            n_rows = other.n_rows;
            n_cols = other.n_cols;

            other.contents = nullptr;
            other.n_rows = 0;
            other.n_cols = 0;
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
     * @param other Matrix to add to this matrix. Must match this matrix's row and column counts.
     */
    template<typename Tp, MatrixStaticType other_static_type>
    void operator+=(const Matrix<Tp, other_static_type>& other) {
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
     * @param other Matrix to subtract from this matrix. Must match this matrix's row and column counts.
     */
    template<typename Tp, MatrixStaticType other_static_type>
    void operator-=(const Matrix<Tp, other_static_type>& other) {
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
        Matrix result(rows(), cols());

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
    Matrix operator/(const T& scalar) const {
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
    template<typename CharT, typename Traits, typename Tp, MatrixStaticType m_static_type>
    friend std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& output_stream, const Matrix<Tp, m_static_type>& matr);

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
template<typename T, MatrixStaticType static_type>
Matrix<T, static_type> operator-(const T& scalar, const Matrix<T, static_type>& matrix)
{
    Matrix<T, static_type> result(matrix.rows(), matrix.cols());

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
template<typename T, MatrixStaticType static_type>
inline Matrix<T, static_type> operator*(const T& scalar, const Matrix<T, static_type>& matrix) {
    Matrix<T, static_type> result(matrix.rows(), matrix.cols());

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
 * The output's static type is always a matrix.
 * 
 * NOTE! Eigen Lite does not support implicit element-wise multiplication!
 * To do element-wise multiplication, use `cwiseProduct`.
 *
 * @param lhs Left-hand side of the multiplication
 * @param rhs Right-hand side in the multiplication. Must satisfy `lhs.cols()` == `rhs.rows()`.
 * @return A new matrix containing the matrix multiplication result.
 */
template<typename T, MatrixStaticType lhs_static_type, MatrixStaticType rhs_static_type>
inline Matrix<T, MATRIX> operator*(const Matrix<T, lhs_static_type>& lhs, const Matrix<T, rhs_static_type>& rhs) {
    assert(lhs.cols() == rhs.rows() && "Matrix multiplication requires left-hand side's columns to equal right side's number of rows");

    Matrix<T, MATRIX> result(lhs.rows(), rhs.cols());

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




template<typename CharT, typename Traits, typename Tp, MatrixStaticType m_static_type>
std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& output_stream, const Matrix<Tp, m_static_type>& matr) {
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
 * A statically typed matrix, of type double.
 * 
 * Imitates Eigen's `MatrixXd`.
 */
typedef Matrix<double, MATRIX> MatrixXd;

/**
 * A statically typed vector, of type double.
 * 
 * Imitates Eigen's `VectorXd`.
 */
typedef Matrix<double, COLUMN_VECTOR> VectorXd;




}