#include "crs_dataset.hpp"
#include "utilities.hpp"

// ============================================================================
// CSR Dataset class
// ============================================================================

// Constructor: builds CSR from edge list, then frees the edge list
CSRDataset::CSRDataset(std::vector<Edge> &&edges, int n_nodes, int delta)
    : n_nodes_(n_nodes), delta_(delta), n_edges_(edges.size()),
      ptr_(n_nodes + 1, 0), A_csr(n_edges_, n_edges_){
  build_csr(std::move(edges));
}

// Alternative constructor: from Python list of lists/tuples
// Each inner list/tuple has exactly 3 integers: [u, v, t]
CSRDataset::CSRDataset(const std::vector<std::vector<int>> &edge_list,
                       int n_nodes, int delta)
    : n_nodes_(n_nodes), delta_(delta), n_edges_(edge_list.size()),
      ptr_(n_nodes + 1, 0), A_csr(n_edges_, n_edges_) {
  std::vector<Edge> edges;
  edges.reserve(n_edges_);
  for (const auto &e : edge_list) {
    if (e.size() != 3) {
      throw std::invalid_argument(
          "Each edge must have exactly 3 elements [u, v, t]");
    }
    edges.push_back({e[0], e[1], e[2]});
  }
  build_csr(std::move(edges));
}

// ========================================================================
// Accessors (interface for transition matrix class)
// ========================================================================
int CSRDataset::num_nodes() const { return n_nodes_; }
int CSRDataset::delta() const { return delta_; }

// ========================================================================
// Debug: return data to Python
// ========================================================================

// Return as Python list of lists: [row_ptr, col_idx, edge_data]
nb::tuple CSRDataset::to_python_lists() const {
  return nb::make_tuple(n_nodes_, n_edges_, delta_, ptr_, src_, dst_, T_, lb_,
                        ub_);
}

std::vector<std::vector<int>> CSRDataset::generate_paths(size_t path_size, size_t num_paths) {
  if (path_size >= 8) {
    path_size = 8;
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

  std::vector<std::vector<int>> path_lst;
  size_t path_count = 0;

  for (size_t k = 0; k < n_edges_; ++k) {
    stack.clear();
    u = src_[k];
    v = dst_[k];
    t = T_[k];
    stack.push_back({u, t, k + 1, k});

  DOWN:
    while (stack.size() < path_size) { // the depth is less than path_size
      Frame &f = stack.back();
      if (f.idx >= f.idx1) { // current node has no more child
        // goto BACKTRACK;
        break;
      } else { // Otherwise, go one step deeper
        v = dst_[f.idx];
        auto lb_idx = lb_[f.idx];
        auto ub_idx = ub_[f.idx];
        if (lb_idx >= ub_idx) { // v is a leaf
          stack.push_back({v, f.t + 1, static_cast<size_t>(ub_idx),
                           static_cast<size_t>(lb_idx)});
        } else {
          stack.push_back({v, T_[lb_idx], static_cast<size_t>(ub_idx),
                           static_cast<size_t>(lb_idx)});
        }
      }
    }

  ++path_count;
  if (path_count <= num_paths) {
    // packed_stack = pack_fstack(stack);
    // path_lst.push_back(unpack_vector(packed_stack, stack.size()));
    path_lst.emplace_back(to_vector(stack));
  } else {
    break;
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
  
  return path_lst;
}
// ========================================================================
// Binary I/O
// ========================================================================
void CSRDataset::save_binary(const std::string &path) const {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("Cannot open file for writing: " + path);
  }

  // Write header
  const char magic[] = "CSR1";
  out.write(magic, 4);

  int32_t version = 1;
  out.write(reinterpret_cast<const char *>(&version), sizeof(version));
  out.write(reinterpret_cast<const char *>(&n_nodes_), sizeof(n_nodes_));
  out.write(reinterpret_cast<const char *>(&n_edges_), sizeof(n_edges_));
  out.write(reinterpret_cast<const char *>(&delta_), sizeof(delta_));

  // Write data
  out.write(reinterpret_cast<const char *>(ptr_.data()),
            ptr_.size() * sizeof(int));
  out.write(reinterpret_cast<const char *>(src_.data()),
            src_.size() * sizeof(int));
  out.write(reinterpret_cast<const char *>(dst_.data()),
            dst_.size() * sizeof(int));
  out.write(reinterpret_cast<const char *>(T_.data()), T_.size() * sizeof(int));

  out.write(reinterpret_cast<const char *>(lb_.data()),
            lb_.size() * sizeof(int));
  out.write(reinterpret_cast<const char *>(ub_.data()),
            ub_.size() * sizeof(int));

  if (!out) {
    throw std::runtime_error("Error writing to file: " + path);
  }
}

CSRDataset CSRDataset::load_binary(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("Cannot open file for reading: " + path);
  }

  // Read header
  char magic[4];
  in.read(magic, 4);
  if (std::string(magic, 4) != "CSR1") {
    throw std::runtime_error("Invalid file format");
  }

  int32_t version;
  in.read(reinterpret_cast<char *>(&version), sizeof(version));
  if (version != 1) {
    throw std::runtime_error("Unsupported file version");
  }

  int n_nodes, n_edges, delta;
  in.read(reinterpret_cast<char *>(&n_nodes), sizeof(n_nodes));
  in.read(reinterpret_cast<char *>(&n_edges), sizeof(n_edges));
  in.read(reinterpret_cast<char *>(&delta), sizeof(delta));

  std::vector<Edge> empty_edges;
  CSRDataset dataset(std::move(empty_edges), n_nodes, delta);
  dataset.n_edges_ = n_edges;
  dataset.ptr_.resize(n_nodes + 1);
  dataset.src_.resize(n_edges);
  dataset.dst_.resize(n_edges);
  dataset.T_.resize(n_edges);
  dataset.lb_.resize(n_edges);
  dataset.ub_.resize(n_edges);

  in.read(reinterpret_cast<char *>(dataset.ptr_.data()),
          (n_nodes + 1) * sizeof(int));
  in.read(reinterpret_cast<char *>(dataset.src_.data()),
          (n_edges) * sizeof(int));
  in.read(reinterpret_cast<char *>(dataset.dst_.data()),
          (n_edges) * sizeof(int));
  in.read(reinterpret_cast<char *>(dataset.T_.data()), (n_edges) * sizeof(int));
  in.read(reinterpret_cast<char *>(dataset.lb_.data()),
          (n_edges) * sizeof(int));
  in.read(reinterpret_cast<char *>(dataset.ub_.data()),
          (n_edges) * sizeof(int));

  if (!in) {
    throw std::runtime_error("Error reading file: " + path);
  }

  return dataset;
}

// Build CSR from edge list
void CSRDataset::build_csr(std::vector<Edge> &&edges) {
  // Sort edges by (u, v, t) for CSR construction
  std::sort(edges.begin(), edges.end(), [](const auto &a, const auto &b) {
    return std::tie(a.u, a.t, a.v) < std::tie(b.u, b.t, b.v);
  });

  src_.reserve(n_edges_);
  dst_.reserve(n_edges_);
  T_.reserve(n_edges_);
  lb_.reserve(n_edges_);
  ub_.reserve(n_edges_);
  for (auto &e : edges) {
    ++ptr_[e.u + 1];

    src_.push_back(e.u);
    dst_.push_back(e.v);
    T_.push_back(e.t);
  }
  std::partial_sum(ptr_.begin(), ptr_.end(), ptr_.begin());

  for (size_t i = 0; i < n_edges_; ++i) {
    auto v = dst_[i];
    auto t = T_[i];
    auto idx0 = ptr_[v];
    auto idx1 = ptr_[v + 1];
    auto it0 = std::upper_bound(T_.begin() + idx0, T_.begin() + idx1, t);
    auto it1 = std::upper_bound(it0, T_.begin() + idx1, t + delta_);
    lb_.push_back(static_cast<int>(it0 - T_.begin()));
    ub_.push_back(static_cast<int>(it1 - T_.begin()));
  }

  std::vector<Eigen::Triplet<double>> tripletList;
  tripletList.reserve(n_edges_);
  for (auto& e: edges) {
    tripletList.emplace_back(e.u, e.v, 1.0);
  }
  A_csr.setFromTriplets(tripletList.begin(), tripletList.end());
  A_csr = A_csr.pruned(1.0, 0.0);
  A_csr.makeCompressed();
  // std::sort(edges.begin(), edges.end(), [](const auto &a, const auto &b) {
  //   return std::tie(a.u, a.v) < std::tie(b.u, b.v);
  // });
  // std::vector<size_t> eigen_ptr(n_nodes_+1);
  // std::vector<uint32_t> col_idx;
  // col_idx.reserve(n_nodes_* n_nodes_/4);

  // ++eigen_ptr[edges[0].u+1];
  // col_idx.push_back(edges[0].v);
  // for (size_t i = 1; i < n_edges_; ++i) {
  //   if (edges[i].u == edges[i-1].u and edges[i].v == edges[i-1].v) {
  //     continue;
  //   }
  //   ++eigen_ptr[edges[i].u+1];
  //   col_idx.push_back(edges[i].v);
  // }
  // edges is now freed (moved from)
}

