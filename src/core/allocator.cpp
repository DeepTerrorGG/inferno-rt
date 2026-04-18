#include "inferno/core/allocator.hpp"
#include <cstdlib>

namespace inferno {
namespace core {

ArenaAllocator::ArenaAllocator(size_t pool_size) 
    : pool_size_(pool_size), offset_(0) {
    memory_pool_ = static_cast<char*>(std::malloc(pool_size));
    if (!memory_pool_) {
        throw std::bad_alloc();
    }
}

ArenaAllocator::~ArenaAllocator() {
    std::free(memory_pool_);
}

void* ArenaAllocator::allocate(size_t size) {
    // 32-byte alignment for SIMD
    size_t remainder = size % 32;
    size_t aligned_size = size + (remainder == 0 ? 0 : 32 - remainder);

    if (offset_ + aligned_size > pool_size_) {
        throw std::bad_alloc(); // Out of memory in arena
    }

    void* ptr = memory_pool_ + offset_;
    offset_ += aligned_size;
    return ptr;
}

void ArenaAllocator::reset() {
    offset_ = 0;
}

} // namespace core
} // namespace inferno

