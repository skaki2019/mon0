#pragma once

#include <nanobind/stl/tuple.h>
#include <nanobind/intrusive/counter.h>
#include <nanobind/intrusive/ref.h>

#include <Eigen/SparseCore>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>




namespace nb = nanobind;
using SparseMatrixCSR = Eigen::SparseMatrix<double, Eigen::ColMajor>;

namespace {
struct Edge {
  int u; // source node
  int v; // destination node
  int t; // timestamp or weight
};
} // namespace

class CSRDataset : public nb::intrusive_base {
public:
  CSRDataset(std::vector<Edge> &&edges, int n_nodes, int delta);
  CSRDataset(const std::vector<std::vector<int>> &edge_list, int n_nodes, int delta);

  int num_nodes() const;
  int delta() const;

  nb::tuple to_python_lists() const;

  void save_binary(const std::string &path) const;
  static CSRDataset load_binary(const std::string &path);

  int n_nodes_;
  int n_edges_;
  int delta_;
  std::vector<int> ptr_;
  std::vector<int> src_;
  std::vector<int> dst_;
  std::vector<int> T_;
  std::vector<int> lb_, ub_;
  void build_csr(std::vector<Edge> &&edges);

  SparseMatrixCSR A_csr;
};