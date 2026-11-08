#pragma once

#include "crs_dataset.hpp"

#include <nanobind/intrusive/counter.h>
#include <nanobind/intrusive/ref.h>

#include <tsl/robin_map.h>
#include <Eigen/SparseCore>

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


struct Frame {
  int u;
  int t;
  size_t idx1;
  size_t idx;
};

uint64_t pack_fstack_le(const std::vector<Frame> &v);
uint64_t pack_fstack(const std::vector<Frame> &v);

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
    prefix_mask_ = ~static_cast<uint64_t>(0) << 8 * (7 - ord_);
    hnode_mask_ = ~static_cast<uint64_t>(0) << 8 * (8 - ord_);
  }

  inline void counts(uint64_t packed_path);
  inline void counts(uint64_t packed_path, size_t offset);
  inline void counts_path(uint64_t, size_t);
  inline double log_probability(uint64_t hnode) const;
  inline double log_probability(uint64_t hnode, size_t offset) const;

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

private:
  nb::ref<CSRDataset> dataset_;
  int ord_;
  tsl::robin_map<uint64_t, CountProb> hash_matrix_;
  uint64_t prefix_mask_, hnode_mask_;
  bool counted_, prob_computed_;
  double bayesian_sc_;
};
