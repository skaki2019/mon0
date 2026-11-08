#include "utilities.hpp"


// Pack vector<int> (elements 0-255) into uint64_t in big-endian byte order.
// v[0] occupies the most-significant byte, v[1] the next, and so on; this is
// equivalent to Python's int.from_bytes(v, 'big').
// Assumes vector size <= 8 (since uint64_t is 64 bits = 8 bytes).
std::vector<int> unpack_vector(uint64_t packed, size_t original_size) {
    std::vector<int> result;
    result.reserve(original_size);

    for (size_t i = 0; i < original_size; ++i) {
        result.push_back(static_cast<int>((packed >> ((7 - i) * 8)) & 0xFF));
    }

    return result;
}


void print_packed_be(uint64_t packed) {
    for (size_t i = 0; i < 8; ++i) {
        std::cout << static_cast<int>((packed >> ((7 - i) * 8)) & 0xFF) << ",";
    }
    std::cout << std::endl;

}

std::vector<int> countRange(const std::vector<int>& vec,
                            int i0, int i1, int K) {
    std::vector<int> freq(K, 0);
    for (int i = i0; i < i1; ++i)
        ++freq[vec[i]];
    return freq;
}