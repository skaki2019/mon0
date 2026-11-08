#pragma once

#include <iostream>
#include <vector>
#include <cstdint>

std::vector<int> unpack_vector(uint64_t packed, size_t original_size);
void print_packed_be(uint64_t packed);

std::vector<int> countRange(const std::vector<int>& vec, int i0, int i1, int K);