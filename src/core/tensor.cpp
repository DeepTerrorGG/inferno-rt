#include "inferno/core/tensor.hpp"

namespace inferno {
namespace core {

Tensor::Tensor(const std::vector<size_t>& shape) : shape_(shape), offset_(0) {
    size_ = std::accumulate(shape.begin(), shape.end(), 1ULL, std::multiplies<size_t>());
    storage_ = std::make_shared<TensorStorage>(size_);
    compute_strides();
}

Tensor::Tensor(std::shared_ptr<TensorStorage> storage, 
               const std::vector<size_t>& shape, 
               const std::vector<size_t>& strides,
               size_t offset)
    : storage_(storage), shape_(shape), strides_(strides), offset_(offset) {
    size_ = std::accumulate(shape.begin(), shape.end(), 1ULL, std::multiplies<size_t>());
}

Tensor::~Tensor() = default;

void Tensor::compute_strides() {
    strides_.resize(shape_.size());
    size_t stride = 1;
    for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i) {
        strides_[i] = stride;
        stride *= shape_[i];
    }
}

float& Tensor::at(const std::vector<size_t>& indices) {
    if (indices.size() != shape_.size()) {
        throw std::invalid_argument("Indices dimensionality must match tensor shape dimensionality");
    }
    size_t ptr_offset = offset_;
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] >= shape_[i]) {
            throw std::out_of_range("Index out of bounds");
        }
        ptr_offset += indices[i] * strides_[i];
    }
    return storage_->data()[ptr_offset];
}

const float& Tensor::at(const std::vector<size_t>& indices) const {
    if (indices.size() != shape_.size()) {
        throw std::invalid_argument("Indices dimensionality must match tensor shape dimensionality");
    }
    size_t ptr_offset = offset_;
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] >= shape_[i]) {
            throw std::out_of_range("Index out of bounds");
        }
        ptr_offset += indices[i] * strides_[i];
    }
    return storage_->data()[ptr_offset];
}

Tensor Tensor::slice(size_t dim, size_t start, size_t end) const {
    if (dim >= shape_.size()) {
        throw std::invalid_argument("Dimension out of bounds");
    }
    if (start >= shape_[dim] || end > shape_[dim] || start >= end) {
        throw std::invalid_argument("Invalid slice range");
    }
    
    std::vector<size_t> new_shape = shape_;
    new_shape[dim] = end - start;
    
    size_t new_offset = offset_ + start * strides_[dim];
    
    return Tensor(storage_, new_shape, strides_, new_offset);
}

Tensor Tensor::broadcast(const std::vector<size_t>& target_shape) const {
    if (target_shape.size() < shape_.size()) {
        throw std::invalid_argument("Target shape must have same or more dimensions");
    }

    std::vector<size_t> new_shape = target_shape;
    std::vector<size_t> new_strides(target_shape.size(), 0);

    // Padding difference
    size_t diff = target_shape.size() - shape_.size();

    for (size_t i = 0; i < target_shape.size(); ++i) {
        if (i < diff) {
            new_strides[i] = 0; // Newly added dimensions will have stride 0 (broadcast)
        } else {
            size_t orig_dim = i - diff;
            if (shape_[orig_dim] == 1 && target_shape[i] > 1) {
                // Dim was 1, broadcast to target_shape[i]
                new_strides[i] = 0;
            } else if (shape_[orig_dim] == target_shape[i]) {
                // Match
                new_strides[i] = strides_[orig_dim];
            } else {
                throw std::invalid_argument("Incompatible dimensions for broadcast");
            }
        }
    }

    return Tensor(storage_, new_shape, new_strides, offset_);
}

bool Tensor::is_contiguous() const {
    size_t expected_stride = 1;
    for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i) {
        if (shape_[i] != 1 && strides_[i] != expected_stride) {
            return false;
        }
        expected_stride *= shape_[i];
    }
    return true;
}

} // namespace core
} // namespace inferno

