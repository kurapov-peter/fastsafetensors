# SPDX-License-Identifier: Apache-2.0

from .. import cpp as fstcpp
from ..common import SafeTensorsMetadata, init_logger, is_gpu_found
from ..frameworks import FrameworkOpBase, TensorBase
from ..st_types import Device, DeviceType, DType
from .base import CopierInterface
from .gds import GdsFileCopier
from .nogds import load_library_func
from .registry import CopierConstructFunc, register_copier_constructor

logger = init_logger(__name__)


def is_hipfile_available() -> bool:
    """Return True if libhipfile.so was loaded by the C++ extension."""
    load_library_func()
    return fstcpp.is_hipfile_found()


class HipFileCopier(GdsFileCopier):
    def _shift_down(self, gbuf: fstcpp.gds_device_buffer, shift: int) -> None:
        if shift <= 0:
            return
        # DMA writes must be complete and visible before the on-device shift.
        self.framework.synchronize(self.device)
        n = self.aligned_length
        buf = self._as_uint8(gbuf.get_base_address(), n)
        span = n - shift
        chunk = min(1 << 30, span)
        tmp_gbuf = self.framework.alloc_tensor_memory(chunk, self.device)
        try:
            tmp = self._as_uint8(tmp_gbuf.get_base_address(), chunk)
            count = 0
            while count < span:
                l = min(chunk, span - count)
                # Stage through tmp: src and dst overlap when shift < l.
                self.framework.copy_tensor(
                    tmp[:l], buf[shift + count : shift + count + l]
                )
                self.framework.copy_tensor(buf[count : count + l], tmp[:l])
                count += l
            self.framework.synchronize(self.device)
        finally:
            self.framework.free_tensor_memory(tmp_gbuf, self.device)

    def _as_uint8(self, ptr: int, n: int) -> TensorBase:
        from ..dlpack import from_cuda_buffer

        dl = from_cuda_buffer(ptr, [n], [1], DType.U8, self.device)
        return self.framework.from_dlpack(dl, self.device, DType.U8)


@register_copier_constructor("hipfile")
def new_hipfile_copier(
    device: Device,
    bbuf_size_kb: int = 16 * 1024,
    max_threads: int = 16,
    **kwargs,
) -> CopierConstructFunc:
    load_library_func()
    device_is_not_cpu = device.type != DeviceType.CPU
    if device_is_not_cpu and not is_gpu_found():
        raise Exception(
            "[FAIL] GPU runtime library not found (expected libamdhip64.so)"
        )
    if not fstcpp.is_hipfile_found():
        raise Exception("[FAIL] libhipfile.so not found; cannot use hipfile copier")

    device_id = device.index if device.index is not None else 0
    reader = fstcpp.gds_file_reader(max_threads, device_is_not_cpu, device_id)

    def construct_copier(
        metadata: SafeTensorsMetadata,
        device: Device,
        framework: FrameworkOpBase,
    ) -> CopierInterface:
        return HipFileCopier(metadata, device, reader, framework)

    return construct_copier
