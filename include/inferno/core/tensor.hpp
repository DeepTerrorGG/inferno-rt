#pragma once

#include <vector>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <numeric>

namespace inferno {
namespace core {

class TensorStorage {
public:
    TensorStorage(size_t size) : size_(size), data_(std::make_unique<float[]>(size)) {}
    ~TensorStorage() = default;

    float* data() { return data_.get(); }
    const float* data() const { return data_.get(); }
    size_t size() const { return size_; }

private:
    size_t size_;
    std::unique_ptr<float[]> data_;
    
    // Prevent copying
    TensorStorage(const TensorStorage&) = delete;
    TensorStorage& operator=(const TensorStorage&) = delete;
};

class Tensor {
public:
    // Create new tensor, allocates storage
    Tensor(const std::vector<size_t>& shape);
    
    // Create a view tensor sharing storage
    Tensor(std::shared_ptr<TensorStorage> storage, 
           const std::vector<size_t>& shape, 
           const std::vector<size_t>& strides,
           size_t offset);

    ~Tensor();

    // Basic accessors
    const std::vector<size_t>& shape() const { return shape_; }
    const std::vector<size_t>& strides() const { return strides_; }
    size_t size() const { return size_; }
    
    // Pointer to underlying data
    float* raw_data() { return storage_->data(); }
    const float* raw_data() const { return storage_->data(); }
    
    size_t offset() const { return offset_; }
    
    // N-dimensional access
    float& at(const std::vector<size_t>& indices);
    const float& at(const std::vector<size_t>& indices) const;

    // View Operations
    Tensor slice(size_t dim, size_t start, size_t end) const;
    Tensor broadcast(const std::vector<size_t>& target_shape) const;
    
    bool is_contiguous() const;

private:
    std::shared_ptr<TensorStorage> storage_;
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    size_t size_;
    size_t offset_; // Offset in elements from the start of the storage
    
    void compute_strides();
};

} // namespace core
} // namespace inferno
