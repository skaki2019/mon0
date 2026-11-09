#include "multi_order_network.hpp"
#include "utilities.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>



namespace nb = nanobind;


// ============================================================================
// MultiOrderNetwork
// ============================================================================
// Manages a collection of TransitionMatrix instances at different Markov orders
// (1st-order, 2nd-order, ..., k-th-order), all sharing the same CSRDataset.
// ============================================================================

void MultiOrderNetwork::build_matrix(size_t path_size) {
  //   size_t path_size = static_cast<size_t>(ord_ + 1);

  std::vector<Frame> stack;
  stack.reserve(path_size);
  std::vector<std::pair<int, int>> path;
  path.reserve(path_size);

  int u, v, t;
  size_t idx0, idx1, idx;
  std::vector<int>::iterator it0, it1, it;
  const size_t PSTACKLISTSIZE = 1024;
  int ic = 0;
  uint64_t packed_stack;

  for (size_t k = 0; k < dataset_->n_edges_; ++k) {
    stack.clear();
    u = dataset_->src_[k];
    v = dataset_->dst_[k];
    t = dataset_->T_[k];
    stack.push_back({u, t, k + 1, k});

  DOWN:
    while (stack.size() < path_size) { // the depth is less than path_size
      Frame &f = stack.back();
      if (f.idx >= f.idx1) { // current node has no more child
        // goto BACKTRACK;
        break;
      } else { // Otherwise, go one step deeper
        v = dataset_->dst_[f.idx];
        auto lb_idx = dataset_->lb_[f.idx];
        auto ub_idx = dataset_->ub_[f.idx];
        if (lb_idx >= ub_idx) { // v is a leaf
          stack.push_back(
              {v, f.t + 1, static_cast<size_t>(ub_idx), static_cast<size_t>(lb_idx)});
        } else {
          stack.push_back({v, dataset_->T_[lb_idx], static_cast<size_t>(ub_idx),
                           static_cast<size_t>(lb_idx)});
        }
      }
    }

    packed_stack = pack_fstack(stack);
    // ++nodes_counts_[stack[0].u];
    for (size_t j = 0; j < max_order_; ++j) {
      matrices_[j]->counts_path(packed_stack, static_cast<size_t>(stack.size()));
    }

  BACKTRACK:
    stack.pop_back();
    while (stack.size() > 0) {
      Frame &f = stack.back();
      if (++f.idx < f.idx1) { // Next node available
        goto DOWN;
      } else {
        stack.pop_back();
      }
    }
  }
}

void MultiOrderNetwork::compute_probability() {
  uint64_t total = std::accumulate(nodes_counts_.begin(), nodes_counts_.end(), 0);
  double inv_total = 1.0 / static_cast<double>(total);
  double log_ml = 0.0;
  for (size_t i=0; i<nodes_log_prob_.size(); ++i) {
    if (nodes_counts_[i] > 0) {
      nodes_log_prob_[i] = log(static_cast<double>(nodes_counts_[i]) * inv_total);
      log_ml += std::lgamma(static_cast<double>(nodes_counts_[i] + 1)) 
            + std::lgamma(static_cast<double>(num_nodes_)) 
            - std::lgamma(static_cast<double>(total + num_nodes_));
    }
  }
  bayesian_sc[0] = log_ml;

  for (size_t j = 0; j < max_order_; ++j) {
    log_ml = matrices_[j]->compute_transition_probability(1000);
    bayesian_sc[j+1] = log_ml;
  }
}

double MultiOrderNetwork::compute_loglikelihood(size_t path_size, size_t ord) {
  if (ord > max_order_) {
    return 0.0;
  }

  std::vector<Frame> stack;
  stack.reserve(path_size);
  std::vector<std::pair<int, int>> path;
  path.reserve(path_size);

  int u, v, t;
  size_t idx0, idx1, idx;
  std::vector<int>::iterator it0, it1, it;
  const size_t PSTACKLISTSIZE = 1024;
  int ic = 0;
  uint64_t packed_stack;
  std::vector<size_t> packed_stack_lst;
  packed_stack_lst.reserve(PSTACKLISTSIZE);

  double log_likelihood = 0.0;

  for (size_t k = 0; k < dataset_->n_edges_; ++k) {
    stack.clear();
    u = dataset_->src_[k];
    v = dataset_->dst_[k];
    t = dataset_->T_[k];
    stack.push_back({u, t, k + 1, k});
    size_t ord_ub = std::min(ord, stack.size()-1);

  DOWN:
    while (stack.size() < path_size) { // the depth is less than path_size
      Frame &f = stack.back();
      if (f.idx >= f.idx1) { // current node has no more child
        // goto BACKTRACK;
        break;
      } else { // Otherwise, go one step deeper
        v = dataset_->dst_[f.idx];
        auto lb_idx = dataset_->lb_[f.idx];
        auto ub_idx = dataset_->ub_[f.idx];
        if (lb_idx >= ub_idx) { // v is a leaf
          stack.push_back(
              {v, f.t + 1, static_cast<size_t>(ub_idx), static_cast<size_t>(lb_idx)});
        } else {
          stack.push_back({v, dataset_->T_[lb_idx], static_cast<size_t>(ub_idx),
                           static_cast<size_t>(lb_idx)});
        }
      }
    }

  packed_stack = pack_fstack(stack);
  log_likelihood += matrix0_->nodes_log_prob_[stack[0].u];
  ord_ub = std::min(ord, stack.size()-1);
  for (size_t j = 0; j < ord_ub; ++j) {
    log_likelihood += matrices_[j]->log_probability(packed_stack);
  }
  for (size_t j = 1; j < path_size-ord; ++j) {
    log_likelihood += matrices_[ord-1]->log_probability(packed_stack, static_cast<size_t>(j));
  }

  BACKTRACK:
    stack.pop_back();
    while (stack.size() > 0) {
      Frame &f = stack.back();
      if (++f.idx < f.idx1) { // Next node available
        goto DOWN;
      } else {
        stack.pop_back();
      }
    }
  }

  return log_likelihood;
}

void MultiOrderNetwork::degrees_of_freedom(int order) {
  dof.resize(order+1);
  std::fill(dof.begin(), dof.end(), 0.0);

  dof[0] = static_cast<double>(dataset_->num_nodes()-1);
  SparseMatrixCSR Ak(dataset_->A_csr);
  for (size_t k=1; k<=order; ++k) {
    const double* values = Ak.valuePtr();
    int nnz = Ak.nonZeros();
    
    // std::accumulate is often optimized by compilers into SIMD instructions
    double dof1 = std::accumulate(values, values + nnz, 0.0);

    int cols = Ak.cols();
    std::vector<int> nnz_per_col(cols, 0);

    const int* outer_idx = Ak.outerIndexPtr();

    // Use std::transform to calculate differences efficiently
    std::transform(
        outer_idx, outer_idx + cols,      // Input start: ptr[0] to ptr[cols-1]
        outer_idx + 1,                    // Input next: ptr[1] to ptr[cols]
        nnz_per_col.begin(),              // Output start
        std::minus<int>()                 // Operation: ptr[i+1] - ptr[i]
    );

    int nnz_row =  std::count_if(nnz_per_col.begin(), nnz_per_col.end(), [](int val) {
        return val > 1; });
    
    dof[k] = dof1 - static_cast<double>(nnz_row);

    if (k != order) {
      Ak = Ak * dataset_->A_csr;
    }
  }
  
  std::partial_sum(dof.begin(), dof.end(), dof_acc.begin());

}
// ============================================================================
// nanobind bindings
// ============================================================================
// NB_MODULE(mylib, m) {
//     // CSRDataset with shared_ptr holder
//     nb::class_<CSRDataset, std::shared_ptr<CSRDataset>>(m, "CSRDataset")
//         .def(nb::init<const std::vector<std::vector<int>>&, int, int>(),
//              nb::arg("edge_list"), nb::arg("n"), nb::arg("delta"))
//         .def("num_nodes", &CSRDataset::num_nodes)
//         .def("delta", &CSRDataset::delta)
//         .def("nnz", &CSRDataset::nnz)
//         .def("to_python_lists", &CSRDataset::to_python_lists)
//         .def("row_ptr_numpy", &CSRDataset::row_ptr_numpy)
//         .def("col_idx_numpy", &CSRDataset::col_idx_numpy)
//         .def("edge_data_numpy", &CSRDataset::edge_data_numpy)
//         .def("save_binary", &CSRDataset::save_binary, nb::arg("path"))
//         .def_static("load_binary", &CSRDataset::load_binary, nb::arg("path"));

//     // TransitionMatrix with shared_ptr holder
//     nb::class_<TransitionMatrix, std::shared_ptr<TransitionMatrix>>(m,
//     "TransitionMatrix")
//         .def(nb::init<std::shared_ptr<CSRDataset>, const std::string&>(),
//              nb::arg("dataset"), nb::arg("config"))
//         .def("compute_probabilities", &TransitionMatrix::compute_probabilities,
//              nb::arg("u"));

//     // MultiOrderNetwork
//     nb::class_<MultiOrderNetwork>(m, "MultiOrderNetwork")
//         .def(nb::init<std::shared_ptr<CSRDataset>, int, const std::string&>(),
//              nb::arg("dataset"),
//              nb::arg("max_order"),
//              nb::arg("base_config") = "uniform",
//              "Build transition matrices for orders 1..max_order sharing the dataset")

//         // Accessors
//         .def("dataset", &MultiOrderNetwork::dataset,
//              "Return the shared CSRDataset")
//         .def("max_order", &MultiOrderNetwork::max_order)
//         .def("num_orders", &MultiOrderNetwork::num_orders)
//         .def("get_matrix", &MultiOrderNetwork::get_matrix,
//              nb::arg("order"),
//              "Return the TransitionMatrix for the given order (1-indexed)")

//         // Interfaces (will raise linker errors until implemented in C++)
//         .def("lodataset_likelihood", &MultiOrderNetwork::lodataset_likelihood,
//              nb::arg("order"), nb::arg("path"),
//              "Log-likelihood of a path under the k-th order model")
//         .def("bayesian_score", &MultiOrderNetwork::bayesian_score,
//              nb::arg("order"), nb::arg("path"),
//              "Bayesian score for the k-th order model")
//         .def("degrees_of_freedom", &MultiOrderNetwork::degrees_of_freedom,
//              nb::arg("order"),
//              "Degrees of freedom of the k-th order model")
//         .def("information_criterion", &MultiOrderNetwork::information_criterion,
//              nb::arg("order"), nb::arg("path"), nb::arg("criterion"),
//              "AIC/BIC/etc. for order selection")
//         .def("optimal_order", &MultiOrderNetwork::optimal_order,
//              nb::arg("path"), nb::arg("criterion"),
//              "Select the best Markov order given a path and criterion")
//         .def("likelihood_ratio_test", &MultiOrderNetwork::likelihood_ratio_test,
//              nb::arg("order_null"), nb::arg("order_alt"), nb::arg("path"),
//              "Likelihood ratio test between two nested orders");
// }