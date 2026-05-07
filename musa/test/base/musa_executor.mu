// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <memory>
#include <type_traits>

#include <gtest/gtest.h>

#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/stream.hpp>

#include "common/cuda_hip/base/executor.hpp.inc"
#include "musa/base/scoped_device_id.hpp"
#include "musa/test/utils.hpp"


namespace {


class ExampleOperation : public gko::Operation {
public:
    explicit ExampleOperation(int& val) : value(val) {}

    void run(std::shared_ptr<const gko::OmpExecutor>) const override
    {
        value = -1;
    }

    void run(std::shared_ptr<const gko::ReferenceExecutor>) const override
    {
        value = -2;
    }

    void run(std::shared_ptr<const gko::HipExecutor>) const override
    {
        value = -3;
    }

    void run(std::shared_ptr<const gko::DpcppExecutor>) const override
    {
        value = -4;
    }

    void run(std::shared_ptr<const gko::CudaExecutor>) const override
    {
        musaGetDevice(&value);
    }

    int& value;
};


class CudaExecutor : public ::testing::Test {
protected:
    CudaExecutor()
        :
#ifdef GKO_TEST_NONDEFAULT_STREAM
          stream(0),
          other_stream(gko::CudaExecutor::get_num_devices() - 1),
#endif
          ref(gko::ReferenceExecutor::create()),
          musa(nullptr),
          musa2(nullptr),
          musa3(nullptr)
    {}

    void SetUp()
    {
        ASSERT_GT(gko::CudaExecutor::get_num_devices(), 0);
#ifdef GKO_TEST_NONDEFAULT_STREAM
        musa = gko::CudaExecutor::create(
            0, ref, std::make_shared<gko::CudaAllocator>(), stream.get());
        musa2 = gko::CudaExecutor::create(
            gko::CudaExecutor::get_num_devices() - 1, ref,
            std::make_shared<gko::CudaAllocator>(), other_stream.get());
        musa3 = gko::CudaExecutor::create(
            0, ref, std::make_shared<gko::CudaUnifiedAllocator>(0),
            stream.get());
#else
        musa = gko::CudaExecutor::create(0, ref);
        musa2 = gko::CudaExecutor::create(
            gko::CudaExecutor::get_num_devices() - 1, ref);
        musa3 = gko::CudaExecutor::create(
            0, ref, std::make_shared<gko::CudaUnifiedAllocator>(0));
#endif
    }

    void TearDown()
    {
        if (musa != nullptr) {
            // ensure that previous calls finished and didn't throw an error
            ASSERT_NO_THROW(musa->synchronize());
        }
    }

#ifdef GKO_TEST_NONDEFAULT_STREAM
    gko::musa_stream stream;
    gko::musa_stream other_stream;
#endif
    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::CudaExecutor> musa;
    std::shared_ptr<gko::CudaExecutor> musa2;
    std::shared_ptr<gko::CudaExecutor> musa3;
};


TEST_F(CudaExecutor, CanInstantiateTwoExecutorsOnOneDevice)
{
    auto musa = gko::CudaExecutor::create(0, ref);
    auto musa2 = gko::CudaExecutor::create(0, ref);

    // We want automatic deinitialization to not create any error
}


TEST_F(CudaExecutor, MasterKnowsNumberOfDevices)
{
    int count = 0;
    musaGetDeviceCount(&count);

    auto num_devices = gko::CudaExecutor::get_num_devices();

    ASSERT_EQ(count, num_devices);
}


TEST_F(CudaExecutor, AllocatesAndFreesMemory)
{
    int* ptr = nullptr;

    ASSERT_NO_THROW(ptr = musa->alloc<int>(2));
    ASSERT_NO_THROW(musa->free(ptr));
}


TEST_F(CudaExecutor, FailsWhenOverallocating)
{
    const gko::size_type num_elems = 1ll << 50;  // 4PB of integers
    int* ptr = nullptr;

    ASSERT_THROW(
        {
            ptr = musa->alloc<int>(num_elems);
            musa->synchronize();
        },
        gko::AllocationError);

    musa->free(ptr);
}


__global__ void check_data(int* data)
{
    if (data[0] != 3 || data[1] != 8) {
        asm("trap;");
    }
}


TEST_F(CudaExecutor, CopiesDataToCuda)
{
    int orig[] = {3, 8};
    auto* copy = musa->alloc<int>(2);

    musa->copy_from(ref, 2, orig, copy);

    check_data<<<1, 1, 0, musa->get_stream()>>>(copy);
    ASSERT_NO_THROW(musa->synchronize());
    musa->free(copy);
}


__global__ void check_data2(int* data)
{
    if (data[0] != 4 || data[1] != 8) {
        asm("trap;");
    }
}


TEST_F(CudaExecutor, CanAllocateOnUnifiedMemory)
{
    int orig[] = {3, 8};
    auto* copy = musa3->alloc<int>(2);

    musa3->copy_from(ref, 2, orig, copy);

    check_data<<<1, 1, 0, musa3->get_stream()>>>(copy);
    ASSERT_NO_THROW(musa3->synchronize());
    copy[0] = 4;
    check_data2<<<1, 1, 0, musa3->get_stream()>>>(copy);
    musa3->free(copy);
}


__global__ void init_data(int* data)
{
    data[0] = 3;
    data[1] = 8;
}

TEST_F(CudaExecutor, CopiesDataFromCuda)
{
    int copy[2];
    auto orig = musa->alloc<int>(2);
    init_data<<<1, 1, 0, musa->get_stream()>>>(orig);

    ref->copy_from(musa, 2, orig, copy);

    EXPECT_EQ(3, copy[0]);
    ASSERT_EQ(8, copy[1]);
    musa->free(orig);
}


/* Properly checks if it works only when multiple GPUs exist */
TEST_F(CudaExecutor, PreservesDeviceSettings)
{
    auto previous_device = gko::CudaExecutor::get_num_devices() - 1;
    GKO_ASSERT_NO_CUDA_ERRORS(musaSetDevice(previous_device));
    auto orig = musa->alloc<int>(2);
    int current_device;
    GKO_ASSERT_NO_CUDA_ERRORS(musaGetDevice(&current_device));
    ASSERT_EQ(current_device, previous_device);

    musa->free(orig);
    GKO_ASSERT_NO_CUDA_ERRORS(musaGetDevice(&current_device));
    ASSERT_EQ(current_device, previous_device);
}


TEST_F(CudaExecutor, RunsOnProperDevice)
{
    int value = -1;

    GKO_ASSERT_NO_CUDA_ERRORS(musaSetDevice(0));
    musa2->run(ExampleOperation(value));

    ASSERT_EQ(value, musa2->get_device_id());
}


TEST_F(CudaExecutor, CopiesDataFromCudaToCuda)
{
    int copy[2];
    auto orig = musa->alloc<int>(2);
    GKO_ASSERT_NO_CUDA_ERRORS(musaSetDevice(0));
    init_data<<<1, 1, 0, musa->get_stream()>>>(orig);

    auto copy_musa2 = musa2->alloc<int>(2);
    musa2->copy_from(musa, 2, orig, copy_musa2);

    // Check that the data is really on GPU2 and ensure we did not cheat
    int value = -1;
    GKO_ASSERT_NO_CUDA_ERRORS(musaSetDevice(musa2->get_device_id()));
    check_data<<<1, 1, 0, musa2->get_stream()>>>(copy_musa2);
    GKO_ASSERT_NO_CUDA_ERRORS(musaSetDevice(0));
    musa2->run(ExampleOperation(value));
    ASSERT_EQ(value, musa2->get_device_id());
    // Put the results on OpenMP and run CPU side assertions
    ref->copy_from(musa2, 2, copy_musa2, copy);
    EXPECT_EQ(3, copy[0]);
    ASSERT_EQ(8, copy[1]);
    musa2->free(copy_musa2);
    musa->free(orig);
}


TEST_F(CudaExecutor, Synchronizes)
{
    // Todo design a proper unit test once we support streams
    ASSERT_NO_THROW(musa->synchronize());
}


TEST_F(CudaExecutor, ExecInfoSetsCorrectProperties)
{
    auto dev_id = musa->get_device_id();
    auto num_sm = 0;
    auto major = 0;
    auto minor = 0;
    auto max_threads_per_block = 0;
    auto warp_size = 0;
    GKO_ASSERT_NO_CUDA_ERRORS(musaDeviceGetAttribute(
        &num_sm, musaDevAttrMultiProcessorCount, dev_id));
    GKO_ASSERT_NO_CUDA_ERRORS(musaDeviceGetAttribute(
        &major, musaDevAttrComputeCapabilityMajor, dev_id));
    GKO_ASSERT_NO_CUDA_ERRORS(musaDeviceGetAttribute(
        &minor, musaDevAttrComputeCapabilityMinor, dev_id));
    GKO_ASSERT_NO_CUDA_ERRORS(musaDeviceGetAttribute(
        &max_threads_per_block, musaDevAttrMaxThreadsPerBlock, dev_id));
    GKO_ASSERT_NO_CUDA_ERRORS(
        musaDeviceGetAttribute(&warp_size, musaDevAttrWarpSize, dev_id));
    auto num_cores = comtert_sm_ver_to_cores(major, minor);

    ASSERT_EQ(musa->get_major_version(), major);
    ASSERT_EQ(musa->get_minor_version(), minor);
    ASSERT_EQ(musa->get_num_multiprocessor(), num_sm);
    ASSERT_EQ(musa->get_warp_size(), warp_size);
    ASSERT_EQ(musa->get_num_warps(), num_sm * (num_cores / warp_size));
    ASSERT_EQ(musa->get_num_warps_per_sm(), num_cores / warp_size);
}


}  // namespace
