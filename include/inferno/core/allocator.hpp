#pragma once

#include <cstddef>
#include <new>

namespace inferno {
namespace core {

// Placeholder for custom ArenaAllocator and AlignedMemory allocator
class ArenaAllocator {
public:
    ArenaAllocator(size_t pool_size);
    ~ArenaAllocator();

    void* allocate(size_t size);
    void reset();

private:
    size_t pool_size_;
    size_t offset_;
    char* memory_pool_;
};

} // namespace core
} // namespace inferno

