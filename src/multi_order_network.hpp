#pragma once

#include "crs_dataset.hpp"
#include "transition_matrix.hpp"

#include <nanobind/intrusive/counter.h>
#include <nanobind/intrusive/ref.h>
#include <tsl/robin_map.h>

#include <vector>
#include <string>
#include <tuple>
#include <numeric>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <cstdint>
#include <iostream>

namespace nb = nanobind;


class MultiOrderNetwork : public nb::intrusive_base {
public:
    // ------------------------------------------------------------------------
    // Construction
    // ------------------------------------------------------------------------
    
    // Build transition matrices for orders 1..max_order, all sharing the dataset
    MultiOrderNetwork(nb::ref<CSRDataset> dataset, int max_order)
                    //   ,const std::string& base_config = "uniform")
        : dataset_(dataset),
          max_order_(max_order),
          matrix0_(new TransitionMatrix0(dataset_)) {
        if (!dataset_) {
            throw std::invalid_argument("dataset must not be null");
        }
        if (max_order < 1) {
            throw std::invalid_argument("max_order must be >= 1");
        }
        num_nodes_ = dataset_->num_nodes();
        matrices_.reserve(max_order);
        for (int k = 1; k <= max_order; ++k) {
            // Each order gets its own TransitionMatrix, sharing the dataset
            matrices_.emplace_back(new TransitionMatrix(dataset_, k));
        }
        // nodes_counts_.resize(num_nodes_);
        // nodes_log_prob_.resize(num_nodes_);
        // std::fill(nodes_counts_.begin(), nodes_counts_.end(), 0);
        // std::fill(nodes_log_prob_.begin(), nodes_log_prob_.end(), 0.0);
        bayesian_sc.resize(max_order_+1);
        std::fill(bayesian_sc.begin(), bayesian_sc.end(), 0.0);
    }

    // ------------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------------
    nb::ref<CSRDataset> dataset() const { return dataset_; }
    int max_order() const { return max_order_; }
    int num_orders() const { return static_cast<int>(matrices_.size()); }

    // Get the transition matrix for a specific order (1-indexed)
    nb::ref<TransitionMatrix> get_matrix(int order) const {
        if (order < 1 || order > max_order_) {
            throw std::out_of_range("order out of range [1, max_order]");
        }
        return matrices_[order - 1];
    }

    void build_matrix(size_t path_size);
    void compute_probability();

    double compute_loglikelihood(size_t, size_t);
    double compute_bayesian(size_t);
    // ------------------------------------------------------------------------
    // Interfaces (declarations only — implementations to be provided later)
    // ------------------------------------------------------------------------

    // Log-likelihood of an observed path under the k-th order model.
    // `path` is a sequence of node indices.
    double log_likelihood(int order,
                          const std::vector<int>& path) const;

    // Bayesian score (e.g., marginal likelihood / model evidence) for the
    // k-th order model given the observed path.
    double bayesian_score(int order,
                          const std::vector<int>& path) const;

    // Number of free parameters (degrees of freedom) of the k-th order model.
    void degrees_of_freedom(int order);

    // ------------------------------------------------------------------------
    // Model selection helpers (interfaces only)
    // ------------------------------------------------------------------------

    // Compute an information criterion (e.g., AIC, BIC) for order selection.
    double information_criterion(int order,
                                 const std::vector<int>& path,
                                 const std::string& criterion) const;

    // Select the optimal Markov order given the observed path and a criterion.
    int optimal_order(const std::vector<int>& path,
                      const std::string& criterion) const;

    // Likelihood ratio test between two nested orders.
    double likelihood_ratio_test(int order_null,
                                 int order_alt,
                                 const std::vector<int>& path) const;


    nb::ref<CSRDataset> dataset_;
    size_t num_nodes_;
    nb::ref<TransitionMatrix0> matrix0_;
    std::vector<nb::ref<TransitionMatrix>> matrices_;  // index = order-1
    size_t max_order_;
    std::vector<uint64_t> nodes_counts_;
    std::vector<double> nodes_log_prob_;
    std::vector<double> dof, dof_acc;
    std::vector<double> bayesian_sc;
};