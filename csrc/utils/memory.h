#pragma once

namespace vllm {
namespace xpu {

enum xpuMemcpyKind { HostToDevice, DeviceToHost, DeviceToDevice };

void xpuMemcpy(void* dst, const void* src, size_t n_bytes, xpuMemcpyKind kind);

void xpuAsyncMemcpy(
    void* dst,
    const void* src,
    size_t n_bytes,
    xpuMemcpyKind kind,
    const void* hctx,
    bool is_pinned);

}  // namespace xpu
}  // namespace vllm
