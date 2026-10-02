#pragma once

#include <vector>
#include <random>
#include <cmath>

namespace matrix_utils {
    namespace generation {

        // -----------------------------------------------------------------
        // Generate 1D Dense Matrix (Row-Major) using formula
        // a_ij, b_ij = (rand - 0.5) * exp(phi * randn)
        // -----------------------------------------------------------------
        template <typename T>
        std::vector<T> generate_matrix(int rows, int cols, T phi = 0.5, int seed = 42) {
            std::mt19937 rng(seed);

            // std::uniform_real_distribution generates in [0, 1).
            // We do 1.0 - val to shift it strictly to (0, 1] as defined in Ozaki I&II papers.
            std::uniform_real_distribution<T> unif(0.0, 1.0);

            // Standard normal distribution (mean 0.0, stddev 1.0)
            std::normal_distribution<T> norm(0.0, 1.0);

            std::vector<T> mat(rows * cols);
            for (int i = 0; i < rows * cols; ++i) {
                T rand_val = static_cast<T>(1.0) - unif(rng);
                T randn_val = norm(rng);

                mat[i] = (rand_val - static_cast<T>(0.5)) * std::exp(phi * randn_val);
            }

            return mat;
        }

        // -----------------------------------------------------------------
        // Generate 1D Dense Matrix (Row-Major) mimicking NN Gradients
        // Uses a Gaussian distribution scaled by 1/sqrt(cols)
        // -----------------------------------------------------------------
        template <typename T>
        std::vector<T> generate_nn_matrix(int rows, int cols, int seed = 42) {
            std::mt19937 rng(seed);

            // Variance scaling to prevent the spectral norm from growing with matrix size
            T stddev = static_cast<T>(1.0) / std::sqrt(static_cast<T>(cols));
            std::normal_distribution<T> norm(0.0, stddev);

            std::vector<T> mat(rows * cols);
            for (int i = 0; i < rows * cols; ++i) {
                mat[i] = norm(rng);
            }

            return mat;
        }

        template <typename T>
        std::vector<T> generate_conditioned_matrix(int rows, int cols, T kappa = 10.0, int seed = 42) {
            std::mt19937 rng(seed);
            std::normal_distribution<T> norm(0.0, 1.0);

            std::vector<T> mat(rows * cols, 0.0);
            int min_dim = std::min(rows, cols);

            // 1. Initialize diagonal matrix Sigma with condition number kappa
            // Singular values decay logarithmically from 1.0 to 1.0 / kappa
            for (int i = 0; i < min_dim; ++i) {
                T exponent = static_cast<T>(i) / static_cast<T>(min_dim - 1);
                mat[i * cols + i] = std::pow(static_cast<T>(1.0) / kappa, exponent);
            }

            // Helper: Generate a random unit vector for reflections
            auto generate_unit_vector = [&](int dim) {
                std::vector<T> v(dim);
                T sq_sum = 0.0;
                for (int i = 0; i < dim; ++i) {
                    v[i] = norm(rng);
                    sq_sum += v[i] * v[i];
                }
                T norm_factor = 1.0 / std::sqrt(sq_sum);
                for (int i = 0; i < dim; ++i) v[i] *= norm_factor;
                return v;
            };

            // 2. Mix using Householder reflections: H = I - 2 * v * v^T
            // 4 mixes is enough to completely spread the energy across the dense matrix
            const int num_mixes = 4;

            for (int m = 0; m < num_mixes; ++m) {

                // Apply Right Reflection (Column mixing): X = X * (I - 2 * v_R * v_R^T)
                std::vector<T> v_R = generate_unit_vector(cols);
                std::vector<T> X_v(rows, 0.0);
                for (int i = 0; i < rows; ++i) {
                    for (int j = 0; j < cols; ++j) {
                        X_v[i] += mat[i * cols + j] * v_R[j];
                    }
                }
                for (int i = 0; i < rows; ++i) {
                    for (int j = 0; j < cols; ++j) {
                        mat[i * cols + j] -= 2.0 * X_v[i] * v_R[j];
                    }
                }

                // Apply Left Reflection (Row mixing): X = (I - 2 * v_L * v_L^T) * X
                std::vector<T> v_L = generate_unit_vector(rows);
                std::vector<T> vT_X(cols, 0.0);
                for (int j = 0; j < cols; ++j) {
                    for (int i = 0; i < rows; ++i) {
                        vT_X[j] += v_L[i] * mat[i * cols + j];
                    }
                }
                for (int i = 0; i < rows; ++i) {
                    for (int j = 0; j < cols; ++j) {
                        mat[i * cols + j] -= 2.0 * v_L[i] * vT_X[j];
                    }
                }
            }

            return mat;
        }

    } // namespace generation
} // namespace matrix_utils