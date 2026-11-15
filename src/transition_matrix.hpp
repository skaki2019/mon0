#pragma once

#include "crs_dataset.hpp"

#include <nanobind/intrusive/counter.h>
#include <nanobind/intrusive/ref.h>

#include <tsl/robin_map.h>
#include <Eigen/SparseCore>
#include <Eigen/Sparse>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>


struct CountProb {
  uint64_t count;
  double prob;
  CountProb() : count(0) {}
};

struct TMatHeader {
  uint32_t magic;
  uint32_t version;
  uint64_t num_entries;
};

struct TMatEntry {
  uint64_t key;
  double prob;
};

class TransitionMatrix0 : public nb::intrusive_base {
public:
  TransitionMatrix0() {}
  TransitionMatrix0(nb::ref<CSRDataset> dataset)
      : dataset_(dataset), n_nodes_(dataset_->n_nodes_),
      nodes_counts_(n_nodes_, 0), 
      nodes_log_prob_(n_nodes_, 0.0),
      counted_(false), prob_computed_(false),
      bayesian_sc_(0.0) {
  }

  void build_matrix();
  nb::ref<CSRDataset> dataset_;
  int n_nodes_;
  std::vector<int> nodes_counts_;
  std::vector<double> nodes_log_prob_;
  bool counted_, prob_computed_;
  double bayesian_sc_;
};

class TransitionMatrix : public nb::intrusive_base {
public:
  TransitionMatrix() {}
  TransitionMatrix(nb::ref<CSRDataset> dataset, const int ord)
      : dataset_(dataset), ord_(ord), counted_(false), prob_computed_(false),
      bayesian_sc_(0.0) {
    n_nodes_ = dataset_->n_nodes_;
    transition_mask_ = ~static_cast<uint64_t>(0) << 8 * (7 - ord_);
    hnode_mask_ = ~static_cast<uint64_t>(0) << 8 * (8 - ord_);
  }

  inline uint64_t node_src(uint64_t hnode) const {
    return hnode_mask_ & hnode;
  }

  inline uint64_t node_dst(uint64_t hnode) const {
    return (transition_mask_ & hnode) << 8;
  }

  void counts(uint64_t packed_path);
  void counts(uint64_t packed_path, size_t offset);
  void counts_path(uint64_t, size_t);
  double log_probability(uint64_t hnode) const;
  double log_probability(uint64_t hnode, size_t offset) const;

  void build_matrix(size_t);
  double compute_transition_probability(size_t);

  static constexpr uint32_t TMAT_MAGIC = 0x544D4154; // "TMAT"
  static constexpr uint32_t TMAT_VERSION = 1;
  static constexpr size_t IO_CHUNK = 1 << 16; // 65,536 entries (~1MB) per chunk

  bool save_tmat_binary(const std::string &path);
  bool load_tmat_binary(const std::string &path);

  std::vector<std::pair<std::vector<int>, uint64_t>>
  get_transition_counts(const tsl::robin_map<uint64_t, CountProb> &table, size_t key_len);

  std::vector<std::pair<std::vector<int>, double>>
  get_transition_probs(const tsl::robin_map<uint64_t, CountProb> &table, size_t key_len);


  nb::ref<CSRDataset> dataset_;
  int ord_;
  int n_nodes_;
  tsl::robin_map<uint64_t, CountProb> hash_matrix_;
  uint64_t transition_mask_, hnode_mask_;
  bool counted_, prob_computed_;
  double bayesian_sc_;
};


struct TransitionGraph : public nb::intrusive_base {
  TransitionGraph(size_t ord) : ord_(ord) {
    transition_mask_ = ~static_cast<uint64_t>(0) << 8 * (7 - ord_);
    hnode_mask_ = ~static_cast<uint64_t>(0) << 8 * (8 - ord_);
  }
  void build_graph(nb::ref<TransitionMatrix> tm);

  inline uint64_t node_src(uint64_t hnode) const {
    return hnode_mask_ & hnode;
  }

  inline uint64_t node_dst(uint64_t hnode) const {
    return (transition_mask_ & hnode) << 8;
  }
  size_t ord_;
  std::vector<uint32_t> nptr;
  std::vector<uint64_t> nodes;   // sorted, includes sink nodes
  std::vector<uint32_t> offset;  // nodes.size() + 1
  std::vector<uint32_t> dst_idx; // E
  std::vector<float> prob;       // E  (or uint32 counts + per-node totals)
  std::vector<uint64_t> counts;

  uint64_t transition_mask_, hnode_mask_;
};