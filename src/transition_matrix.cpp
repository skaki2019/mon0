#include "transition_matrix.hpp"
#include "utilities.hpp"

struct Entry {
  uint64_t key;
  CountProb *val_ptr; // Pointer to value in the hash map
};

uint64_t pack_fstack_le(const std::vector<Frame>& v) {
    uint64_t packed = 0;
    for (size_t i = 0; i < v.size(); ++i) {
        packed |= (static_cast<uint64_t>(v[i].u) & 0xFF) << (i * 8);
    }
    return packed;
}

uint64_t pack_fstack(const std::vector<Frame>& v) {
    uint64_t packed = 0;
    const size_t n = v.size();
    for (size_t i = 0; i < n; ++i) {
        packed |= (static_cast<uint64_t>(v[i].u) & 0xFF) << ((7- i) * 8);
    }
    return packed;
}


void TransitionMatrix0::build_matrix() {
  uint64_t total = dataset_->ptr_[n_nodes_];
  std::vector<int> nodes_count(dataset_->ptr_.size());
  std::adjacent_difference(dataset_->ptr_.begin(), dataset_->ptr_.end(), nodes_count.begin());
  std::copy(nodes_count.begin()+1, nodes_count.end()+1, nodes_counts_.begin());
  double inv_total = 1.0 / static_cast<double>(total);
  double log_ml = 0.0;
  for (size_t i=0; i<n_nodes_; ++i) {
    if (nodes_counts_[i] > 0) {
      nodes_log_prob_[i] = log(static_cast<double>(nodes_counts_[i]) * inv_total);
      log_ml += std::lgamma(static_cast<double>(nodes_counts_[i] + 1)) 
            + std::lgamma(static_cast<double>(n_nodes_)) 
            - std::lgamma(static_cast<double>(total + n_nodes_));
    }
  }
  bayesian_sc_ = log_ml;
  counted_ = true;
  prob_computed_ = true;
}

void TransitionMatrix::counts(uint64_t packed_path) {
  uint64_t key = packed_path & prefix_mask_;
  auto [it, inserted] = hash_matrix_.try_emplace(key, CountProb{});
  ++(it.value().count);
}

void TransitionMatrix::counts(uint64_t packed_path, size_t offset) {
  uint64_t key = (packed_path << 8*offset) & prefix_mask_;
  auto [it, inserted] = hash_matrix_.try_emplace(key, CountProb{});
  ++(it.value().count);
}

void TransitionMatrix::counts_path(uint64_t packed_path, size_t path_size) {
  for (size_t j = 0; j < path_size-ord_; ++j) {
    counts(packed_path, j);
  }
}

double TransitionMatrix::log_probability(uint64_t packed_path) const{
  uint64_t key = packed_path & prefix_mask_;
  auto it = hash_matrix_.find(key);
  if (it != hash_matrix_.end()) {
    return it->second.prob;
  } else {
    return 0.0;
  }
  // return hash_matrix_[key].prob;
}

double TransitionMatrix::log_probability(uint64_t packed_path, size_t offset) const{
  uint64_t key = (packed_path << 8*offset) & prefix_mask_;
  auto it = hash_matrix_.find(key);
  if (it != hash_matrix_.end()) {
    return it->second.prob;
  } else {
    return 0.0;
  }
}
// ============================================================================
// Transition Matrix class (example usage of CSRDataset interface)
// ============================================================================
void TransitionMatrix::build_matrix(size_t path_size) {
  if (counted_) {
    return;
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
          stack.push_back({v, f.t + 1, static_cast<size_t>(ub_idx),
                           static_cast<size_t>(lb_idx)});
        } else {
          stack.push_back({v, dataset_->T_[lb_idx], static_cast<size_t>(ub_idx),
                           static_cast<size_t>(lb_idx)});
        }
      }
    }

  packed_stack = pack_fstack(stack);
  for (size_t j = 0; j < stack.size()-ord_; ++j) {
    counts(packed_stack, static_cast<size_t>(j));
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

  counted_ = true;
}

double TransitionMatrix::compute_transition_probability(size_t msize=1'000'000) {
  if (prob_computed_) {
    return bayesian_sc_;
  }
  if (hash_matrix_.empty()) {
    return 0.0;
  }

  if (ord_ == 0) {
    uint64_t total = 0;
    for (auto& pair: hash_matrix_) {
      total += pair.second.count;
    }
    for (auto it=hash_matrix_.begin(); it != hash_matrix_.end(); ++it) {
      it.value().prob = static_cast<double>(it->second.count) / static_cast<double>(total);
    }
    return 0.0;
  }

  double log_ml = 0.0;
  auto process_batch = [&](std::vector<Entry> &batch) {
    if (batch.empty())
      return;

    // 1. Sort by packed key (lexicographic order preserved by integer
    // comparison)
    std::sort(batch.begin(), batch.end(),
              [](const Entry &a, const Entry &b) { return a.key < b.key; });

    // 2. Two-pass grouping in a single linear scan
    size_t i = 0;
    const size_t n = batch.size();
    while (i < n) {
      size_t j = i;
      uint64_t current_hnode = batch[i].key & hnode_mask_;
      uint64_t total = 0;
      size_t num_nodes = 0;

      // --- Pass 1: Accumulate total for this prefix ---
      while (j < n && (batch[j].key & hnode_mask_) == current_hnode) {
        total += batch[j].val_ptr->count;
        ++num_nodes;
        ++j;
      }

      // --- Pass 2: Normalize and write back ---
      if (total > 0) {
        double inv_total = 1.0 / static_cast<double>(total);
        for (size_t k = i; k < j; ++k) {
          uint64_t c = batch[k].val_ptr->count;
          log_ml += std::lgamma(static_cast<double>(c + 1)) 
            + std::lgamma(static_cast<double>(num_nodes)) 
            - std::lgamma(static_cast<double>(total + num_nodes));
          batch[k].val_ptr->prob = log(static_cast<double>(c) * inv_total);
        }
      }

      // Move to next group
      i = j;
    }
  };

  // --- Strategy: Chunk if the size of hash_matrix_ is large
  const bool use_chunking = hash_matrix_.size() > msize;
  const size_t end_chunk = use_chunking ? 16 : 1;

  std::vector<Entry> batch;
  batch.reserve(use_chunking? hash_matrix_.size() / 8 : hash_matrix_.size());

  for (size_t chunk = 0; chunk < end_chunk; ++chunk) {
    // Extract matching keys
    for (auto &pair : hash_matrix_) {
      uint64_t first_byte = pair.first >> 56;
      if (!use_chunking || first_byte%16 == chunk) {
        batch.push_back({pair.first, const_cast<CountProb *>(&pair.second)});
      }
    }

    if (!batch.empty()) {
      process_batch(batch);
    }
    batch.clear();
  }

  bayesian_sc_ = log_ml;
  prob_computed_ = true;
  return log_ml;
}

bool TransitionMatrix::save_tmat_binary(const std::string &path) {
  std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
  if (!ofs)
    return false;

  TMatHeader header{TMAT_MAGIC, TMAT_VERSION, hash_matrix_.size()};
  ofs.write(reinterpret_cast<const char *>(&header), sizeof(header));
  if (!ofs)
    return false;

  std::vector<TMatEntry> buffer;
  buffer.reserve(IO_CHUNK);

  for (const auto &[key, val] : hash_matrix_) {
    buffer.push_back({key, val.prob});
    if (buffer.size() >= IO_CHUNK) {
      ofs.write(reinterpret_cast<const char *>(buffer.data()),
                sizeof(TMatEntry) * buffer.size());
      if (!ofs)
        return false;
      buffer.clear();
    }
  }
  // Flush remaining
  if (!buffer.empty()) {
    ofs.write(reinterpret_cast<const char *>(buffer.data()),
              sizeof(TMatEntry) * buffer.size());
  }
  return ofs.good();
}

bool TransitionMatrix::load_tmat_binary(const std::string &path) {
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs)
    return false;

  TMatHeader header;
  ifs.read(reinterpret_cast<char *>(&header), sizeof(header));
  if (!ifs || header.magic != TMAT_MAGIC || header.version != TMAT_VERSION) {
    return false; // Invalid or corrupted file
  }

  hash_matrix_.clear();
  hash_matrix_.reserve(header.num_entries); // Avoids rehashing during insertion

  std::vector<TMatEntry> buffer(IO_CHUNK);
  uint64_t remaining = header.num_entries;

  while (remaining > 0) {
    size_t to_read = std::min(IO_CHUNK, static_cast<size_t>(remaining));
    ifs.read(reinterpret_cast<char *>(buffer.data()),
             sizeof(TMatEntry) * to_read);
    if (!ifs)
      return false;

    for (size_t i = 0; i < to_read; ++i) {
      CountProb val;
      val.prob = buffer[i].prob;
      hash_matrix_.emplace(buffer[i].key, val);
    }
    remaining -= to_read;
  }

  counted_ = true;
  prob_computed_ = true;
  return true;
}

// 2. Debug Export: Counts
std::vector<std::pair<std::vector<int>, uint64_t>>
TransitionMatrix::get_transition_counts(
    const tsl::robin_map<uint64_t, CountProb> &table, size_t key_len) {
  std::vector<std::pair<std::vector<int>, uint64_t>> result;
  result.reserve(table.size());

  for (const auto &[key, val] : table) {
    result.emplace_back(unpack_vector(key, key_len), val.count);
  }
  return result;
}

// 3. Debug Export: Probabilities
std::vector<std::pair<std::vector<int>, double>>
TransitionMatrix::get_transition_probs(
    const tsl::robin_map<uint64_t, CountProb> &table, size_t key_len) {
  std::vector<std::pair<std::vector<int>, double>> result;
  result.reserve(table.size());

  for (const auto &[key, val] : table) {
    result.emplace_back(unpack_vector(key, key_len), val.prob);
  }
  return result;
}
