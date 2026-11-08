// CPU-only transfer mock for the actual snapshot/file routines. Never link this
// into an engine. No Metal framework, GPU runtime or model is initialized.
#include "strata/gpu/runtime.hpp"
#include <cstdlib>
#include <cstring>
#if !defined(SUMMARY_DISK_TEST_GPU)
namespace strata::gpu {
void* host_view(const void* p) { return const_cast<void*>(p); }
Error get_last_error() { return kSuccess; }
Error peek_last_error() { return kSuccess; }
const char* error_string(Error) { return "CPU mock transfer"; }
Error get_device(int* d) { *d=0; return kSuccess; }
Error set_device(int) { return kSuccess; }
Error device_count(int* n) { *n=1; return kSuccess; }
Error device_synchronize() { return kSuccess; }
Error mem_get_info(size_t* f,size_t* t) { *f=*t=size_t(1)<<30; return kSuccess; }
Error malloc_device_raw(void** p,size_t n) { *p=std::malloc(n);return *p?kSuccess:kErrorMemoryAllocation; }
Error free_device(void* p) { std::free(p);return kSuccess; }
Error host_alloc_raw(void** p,size_t n,unsigned) { return malloc_device_raw(p,n); }
Error free_host(void* p) { return free_device(p); }
Error host_device_pointer_raw(void** p,void* h) { *p=h;return kSuccess; }
Error memset_device(void* p,int v,size_t n) { std::memset(p,v,n);return kSuccess; }
Error memset_async(void* p,int v,size_t n,Stream) { return memset_device(p,v,n); }
Error memcpy_sync(void* d,const void* s,size_t n,CopyKind) {
    if(n && (!d || !s)) return kErrorInvalidValue;
    if(n) std::memcpy(d,s,n);return kSuccess;
}
Error memcpy_async(void* d,const void* s,size_t n,CopyKind k,Stream) { return memcpy_sync(d,s,n,k); }
Error stream_create(Stream* s,unsigned) { *s=reinterpret_cast<Stream>(1);return kSuccess; }
Error stream_destroy(Stream) { return kSuccess; }
Error stream_synchronize(Stream) { return kSuccess; }
Error stream_query(Stream) { return kSuccess; }
Error stream_wait_event(Stream,Event,unsigned) { return kSuccess; }
Error event_create(Event* e,unsigned) { *e=reinterpret_cast<Event>(1);return kSuccess; }
Error event_destroy(Event) { return kSuccess; }
Error event_record(Event,Stream) { return kSuccess; }
Error event_query(Event) { return kSuccess; }
Error event_synchronize(Event) { return kSuccess; }
}
#endif
#include "strata/core/layer.hpp"
#include "strata/kernels/kv_stream.hpp"
namespace strata::core {
strata::kernels::QsaAttnPools qsa_attn_pools(const QsaState&) { std::abort(); }
}
namespace strata::kernels {
void kv_stream_reset(const KvStreamMap&,void*) { std::abort(); }
void kv_ring_restore(const QsaAttnPools&,const KvHostPools&,int,int64_t,int64_t,int64_t,const QsaShapes&,void*) { std::abort(); }
}
