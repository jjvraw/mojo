# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
#
# Tests that _cublas_matmul passes stride[0] as lda/ldb rather than the inner
# dimension, so strided (non-contiguous) TileTensors produce correct results.

from std.math import ceildiv
from std.random import random_float64
from std.memory import UnsafePointer

import linalg.matmul.vendor.blas as vendor_blas
from std.gpu.host import DeviceContext
from layout import Coord, MixedLayout, TileTensor, row_major
from linalg.matmul.gpu import matmul_kernel_naive
from std.testing import assert_almost_equal


def test_vendor_blas_strided[
    dtype: DType, M: Int, N: Int, K: Int, lda_pad: Int, ldb_pad: Int
](ctx: DeviceContext) raises:
    comptime lda = K + lda_pad
    comptime ldb = N + ldb_pad
    print(
        "== test_vendor_blas_strided",
        dtype,
        "M=", M, "N=", N, "K=", K,
        "lda=", lda, "ldb=", ldb,
    )

    var a_host = ctx.enqueue_create_host_buffer[dtype](M * lda)
    var b_host = ctx.enqueue_create_host_buffer[dtype](K * ldb)
    var c_host = ctx.enqueue_create_host_buffer[dtype](M * N)
    var c_host_ref = ctx.enqueue_create_host_buffer[dtype](M * N)

    for m in range(M):
        for k in range(K):
            a_host[m * lda + k] = random_float64(-0.1, 0.1).cast[dtype]()
    for k in range(K):
        for n in range(N):
            b_host[k * ldb + n] = random_float64(-0.1, 0.1).cast[dtype]()

    var a_device = ctx.enqueue_create_buffer[dtype](M * lda)
    var b_device = ctx.enqueue_create_buffer[dtype](K * ldb)
    var c_device = ctx.enqueue_create_buffer[dtype](M * N)
    var c_device_ref = ctx.enqueue_create_buffer[dtype](M * N)

    ctx.enqueue_copy(a_device, a_host)
    ctx.enqueue_copy(b_device, b_host)

    var a = TileTensor(a_device, MixedLayout(Coord(M, K), Coord(lda, 1)))
    var b = TileTensor(b_device, MixedLayout(Coord(K, N), Coord(ldb, 1)))
    var c = TileTensor(c_device, row_major(Coord(M, N)))

    vendor_blas.matmul(ctx, c, a, b, c_row_major=True, transpose_b=False)

    ctx.enqueue_copy(c_host, c_device)

    comptime BLOCK_DIM = 16

    var c_ref_tt = TileTensor(c_device_ref, row_major(Coord(M, N)))
    var a_tt = TileTensor(
        UnsafePointer[Scalar[dtype], ImmutAnyOrigin](
            unsafe_from_address=Int(a_device.unsafe_ptr())
        ),
        MixedLayout(Coord(M, K), Coord(lda, 1)),
    )
    var b_tt = TileTensor(
        UnsafePointer[Scalar[dtype], ImmutAnyOrigin](
            unsafe_from_address=Int(b_device.unsafe_ptr())
        ),
        MixedLayout(Coord(K, N), Coord(ldb, 1)),
    )

    comptime kernel = matmul_kernel_naive[
        dtype, dtype, dtype,
        type_of(c_ref_tt).LayoutType,
        type_of(a_tt).LayoutType,
        type_of(b_tt).LayoutType,
        BLOCK_DIM,
        transpose_b=False,
    ]
    ctx.enqueue_function[kernel](
        c_ref_tt, a_tt, b_tt, M, N, K,
        grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM), 1),
        block_dim=(BLOCK_DIM, BLOCK_DIM, 1),
    )

    ctx.enqueue_copy(c_host_ref, c_device_ref)
    ctx.synchronize()

    for i in range(M * N):
        assert_almost_equal(
            c_host[i],
            c_host_ref[i],
            atol=1e-2 if dtype.is_half_float() else 1e-3,
            rtol=1e-2 if dtype.is_half_float() else 1e-3,
        )

    _ = a_device
    _ = b_device
    _ = c_device
    _ = c_device_ref


def main() raises:
    with DeviceContext() as ctx:
        test_vendor_blas_strided[DType.float32, 64, 64, 64, lda_pad=16, ldb_pad=8](ctx)
        test_vendor_blas_strided[DType.bfloat16, 128, 256, 128, lda_pad=32, ldb_pad=0](ctx)
        test_vendor_blas_strided[DType.float32, 63, 65, 66, lda_pad=2, ldb_pad=3](ctx)

        test_vendor_blas_strided[DType.float32, 64, 64, 64, lda_pad=0, ldb_pad=0](ctx)
        test_vendor_blas_strided[DType.bfloat16, 512, 512, 512, lda_pad=0, ldb_pad=0](ctx)
