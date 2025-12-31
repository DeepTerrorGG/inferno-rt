#include <gtest/gtest.h>
#include "inferno/core/tensor.hpp"

using namespace inferno::core;

TEST(TensorTest, BasicInitialization) {
    Tensor t({2, 3, 4});
    EXPECT_EQ(t.shape().size(), 3);
    EXPECT_EQ(t.size(), 24);
    EXPECT_NE(t.raw_data(), nullptr);
}

TEST(TensorTest, StrideCalculation) {
    Tensor t({2, 3, 4});
    const auto& strides = t.strides();
    EXPECT_EQ(strides.size(), 3);
    EXPECT_EQ(strides[2], 1);
    EXPECT_EQ(strides[1], 4);
    EXPECT_EQ(strides[0], 12);
}

TEST(TensorTest, ElementAccess) {
    Tensor t({2, 3});
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            t.at({i, j}) = static_cast<float>(i * 3 + j);
        }
    }
    EXPECT_FLOAT_EQ(t.at({0, 0}), 0.0f);
    EXPECT_FLOAT_EQ(t.at({0, 1}), 1.0f);
    EXPECT_FLOAT_EQ(t.at({1, 2}), 5.0f);
}

TEST(TensorTest, Slicing) {
    Tensor t({3, 3});
    for(size_t i=0; i<3; ++i) {
        for(size_t j=0; j<3; ++j) {
            t.at({i, j}) = static_cast<float>(i * 3 + j);
        }
    }

    Tensor row_slice = t.slice(0, 1, 2);
    EXPECT_EQ(row_slice.shape().size(), 2);
    EXPECT_EQ(row_slice.shape()[0], 1);
    EXPECT_EQ(row_slice.shape()[1], 3);

    EXPECT_FLOAT_EQ(row_slice.at({0, 0}), 3.0f);
    EXPECT_FLOAT_EQ(row_slice.at({0, 1}), 4.0f);
    EXPECT_FLOAT_EQ(row_slice.at({0, 2}), 5.0f);

    row_slice.at({0, 1}) = 99.0f;
    EXPECT_FLOAT_EQ(t.at({1, 1}), 99.0f);
    
    EXPECT_TRUE(row_slice.is_contiguous());

    Tensor col_slice = t.slice(1, 1, 2);
    EXPECT_FALSE(col_slice.is_contiguous());
    EXPECT_FLOAT_EQ(col_slice.at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(col_slice.at({1, 0}), 99.0f);
    EXPECT_FLOAT_EQ(col_slice.at({2, 0}), 7.0f);
}

TEST(TensorTest, Broadcasting) {
    Tensor t({1, 3});
    t.at({0, 0}) = 1.0f;
    t.at({0, 1}) = 2.0f;
    t.at({0, 2}) = 3.0f;

    Tensor broadcasted = t.broadcast({4, 3});
    EXPECT_EQ(broadcasted.shape()[0], 4);
    EXPECT_EQ(broadcasted.shape()[1], 3);
    EXPECT_EQ(broadcasted.strides()[0], 0);

    for(size_t i=0; i<4; ++i) {
        EXPECT_FLOAT_EQ(broadcasted.at({i, 0}), 1.0f);
        EXPECT_FLOAT_EQ(broadcasted.at({i, 1}), 2.0f);
        EXPECT_FLOAT_EQ(broadcasted.at({i, 2}), 3.0f);
    }
}
