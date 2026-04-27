#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>
#include <vector>

namespace py = pybind11;


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    for (size_t i = 0; i < m; i += batch) {
        const size_t batch_size = std::min(batch, m - i);
        std::vector<float> probs(batch_size * k, 0.0f);
        std::vector<float> grad(n * k, 0.0f);

        // probs[b, c] = softmax((X_batch @ theta)[b, c])
        // Compute one sample's logits by accumulating weighted rows of theta:
        // score[c] = sum_j X[b, j] * theta[j, c]
        for (size_t b = 0; b < batch_size; ++b) {
            const size_t x_row = (i + b) * n;
            float *prob_row = &probs[b * k];

            for (size_t j = 0; j < n; ++j) {
                const float x_val = X[x_row + j];
                const size_t theta_row = j * k;
                for (size_t c = 0; c < k; ++c) {
                    prob_row[c] += x_val * theta[theta_row + c];
                }
            }

            float sum_exp = 0.0f;
            for (size_t c = 0; c < k; ++c) {
                const float ex = std::exp(prob_row[c]);
                prob_row[c] = ex;
                sum_exp += ex;
            }
            for (size_t c = 0; c < k; ++c) {
                prob_row[c] /= sum_exp;
            }
            prob_row[y[i + b]] -= 1.0f;
        }

        // grad = X_batch^T @ (probs - one_hot) / batch_size
        for (size_t j = 0; j < n; ++j) {
            for (size_t c = 0; c < k; ++c) {
                float acc = 0.0f;
                for (size_t b = 0; b < batch_size; ++b) {
                    acc += X[(i + b) * n + j] * probs[b * k + c];
                }
                grad[j * k + c] = acc / static_cast<float>(batch_size);
            }
        }

        for (size_t idx = 0; idx < n * k; ++idx) {
            theta[idx] -= lr * grad[idx];
        }
    }
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
