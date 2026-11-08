// Real snapshot/source/file/read-limit/restore code with host-memory transfers.
// This does not qualify Metal kernels, GPU synchronization, or model output.
#include "strata/core/conversation_snapshot.hpp"
#include <cstdio>
#include <chrono>
#include <cuda_runtime.h>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
using namespace strata::core;
static int checks=0;
static void check(bool ok,const char* label,const std::string& error={}) {
    ++checks;if(!ok){std::fprintf(stderr,"FAIL: %s: %s\n",label,error.c_str());std::exit(1);}
}
struct Pool {
    QsaState st;
    std::array<std::vector<uint8_t>,8> b;
#if defined(SUMMARY_DISK_TEST_GPU)
    std::array<void*,8> device{};
    ~Pool() { for (auto p : device) if (p) cudaFree(p); }
#endif
    explicit Pool(const ModelGeometry& g) {
        st.max_cells=96;st.n_pages=st.n_slots=24;st.idx_pooled_rows=26;st.kv_int8=true;
        b[0].resize(96*g.n_head_kv*g.head_dim,13);b[1]=b[0];
        b[2].resize(96*g.n_head_kv*(g.head_dim/64)*2,13);b[3]=b[2];
        b[4].resize(26*g.idx_key_dim*4,13);b[5].resize(3*g.idx_key_dim*4,13);
        b[6].resize(g.idx_key_dim*4,13);b[7].resize(4,0);
        auto ptr = [&](size_t i) -> void* {
#if defined(SUMMARY_DISK_TEST_GPU)
            check(cudaMalloc(&device[i],b[i].size())==cudaSuccess,"tiny GPU pool allocation");
            check(cudaMemcpy(device[i],b[i].data(),b[i].size(),cudaMemcpyHostToDevice)==cudaSuccess,"tiny GPU pool upload");
            return device[i];
#else
            return b[i].data();
#endif
        };
        st.k_q=(int8_t*)ptr(0);st.v_q=(int8_t*)ptr(1);
        st.k_scale=(uint16_t*)ptr(2);st.v_scale=(uint16_t*)ptr(3);
        st.idx_pooled=(float*)ptr(4);st.idx_tail=(float*)ptr(5);
        st.idx_dead=(float*)ptr(6);st.idx_block_pos=(int32_t*)ptr(7);
    }
};
static std::vector<char> read(const std::string& p) {
    std::ifstream f(p,std::ios::binary);return {std::istreambuf_iterator<char>(f),{}};
}
int main() {
    ModelGeometry g;g.n_layers=4;g.ssm_state_size=2;g.ssm_v_heads=2;g.ssm_conv_channels=8;
    g.n_head_kv=1;g.head_dim=64;g.idx_key_dim=8;
    ConversationStateSizes z;std::string error;
    check(conversation_state_sizes(g,z,error),"fixture geometry",error);
    Pool main(g),draft(g);std::vector<uint8_t> state(z.gdn,13),ple(z.ple,13);
    SessionState ss;ss.max_cells=96;ss.layer_hi=4;ss.gdn_alloc=g.n_gdn_layers();ss.qsa_alloc=g.n_qsa_layers();
    ss.qsa_states=&main.st;ss.gdn_state=(float*)state.data();ss.ple_hist=(float*)ple.data();
#if defined(SUMMARY_DISK_TEST_GPU)
    check(cudaMalloc(&ss.gdn_state,state.size())==cudaSuccess && cudaMalloc(&ss.ple_hist,ple.size())==cudaSuccess,"tiny GPU running-state allocation");
    check(cudaMemcpy(ss.gdn_state,state.data(),state.size(),cudaMemcpyHostToDevice)==cudaSuccess &&
          cudaMemcpy(ss.ple_hist,ple.data(),ple.size(),cudaMemcpyHostToDevice)==cudaSuccess,"tiny GPU running-state upload");
#endif
    std::vector<int32_t> ids{1,2,3,4,5,6,7,8,9};std::vector<ConversationImageKey> imgs;std::vector<ConversationCheckpoint> ckpts;
    const ConversationView view{ids,imgs,ckpts,true};
    QsaState absent;SavedConversation meta;std::vector<SessionKvSource> sources;
    check(!conversation_snapshot_sources(meta,sources,view,ss,g,absent,error) && error=="conversation snapshot: invalid K/V extent",
          "reproduce old no-MTP caller passing dummy draft",error);
    namespace fs=std::filesystem;
    const auto dir=fs::temp_directory_path()/("strata-summary-cpu-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(dir);const SessionFileIdentity identity{33,44};
    for(bool mtp:{false,true}) {
        const QsaState* d=mtp?&draft.st:nullptr;
        check(conversation_snapshot_sources(meta,sources,view,ss,g,d,error),"nonempty streamed sources",error);
        check(sources.size()==size_t(ss.qsa_alloc)+(mtp?1:0),"exact optional draft layer count");
        SessionReadLimits limits;
        check(conversation_session_read_limits(limits,ss,g,d,96,6,error),"nullable read limits",error);
        check(limits.max_kv_layers==sources.size() && limits.max_kv_bytes.size()==sources.size(),"read limits match sources");
        const auto path=(dir/(mtp?"mtp.bin":"target.bin")).string();size_t bytes=0;
        check(session_file_write(path,meta,sources,identity,bytes,error),"write actual streamed file",error);
        check(bytes>0,"nonempty snapshot bytes");
        SavedConversation image;size_t read_bytes=0;
        check(session_file_read(path,identity,image,read_bytes,error,limits),"read bounded actual file",error);
        check(read_bytes==bytes && image.kv.size()==sources.size(),"roundtrip file count/bytes");
        check(conversation_snapshot_validate(image,ss,g,d,error),"validate optional draft image",error);
        check(!conversation_snapshot_validate(image,ss,g,mtp?nullptr:&draft.st,error),"reject wrong draft presence");
        std::fill(state.begin(),state.end(),99);std::fill(main.b[0].begin(),main.b[0].end(),99);
#if defined(SUMMARY_DISK_TEST_GPU)
        check(cudaMemcpy(ss.gdn_state,state.data(),state.size(),cudaMemcpyHostToDevice)==cudaSuccess &&
              cudaMemcpy(main.st.k_q,main.b[0].data(),main.b[0].size(),cudaMemcpyHostToDevice)==cudaSuccess,"poison tiny GPU fixture before restore");
#endif
        check(conversation_snapshot_restore(image,ss,g,d,error)==ConversationRestore::restored,"restore through actual production routine",error);
#if defined(SUMMARY_DISK_TEST_GPU)
        check(cudaMemcpy(state.data(),ss.gdn_state,state.size(),cudaMemcpyDeviceToHost)==cudaSuccess &&
              cudaMemcpy(main.b[0].data(),main.st.k_q,main.b[0].size(),cudaMemcpyDeviceToHost)==cudaSuccess,"read restored tiny GPU fixture");
#endif
        check(state.front()==13 && main.b[0].front()==13 && ss.ple_prev[0]==8 && ss.ple_prev[1]==9,"restored recurrent/KV/token state");
        if(mtp) {
            SavedConversation ref;std::vector<SessionKvSource> rs;
            check(conversation_snapshot_sources(ref,rs,view,ss,g,draft.st,error),"legacy MTP reference source overload",error);
            const auto legacy=(dir/"legacy.bin").string();size_t n=0;
            check(session_file_write(legacy,ref,rs,identity,n,error),"legacy MTP file",error);
            check(read(legacy)==read(path),"MTP reference and nullable files byte identical");
            SessionReadLimits rl;check(conversation_session_read_limits(rl,ss,g,draft.st,96,6,error),"legacy MTP limit overload",error);
            check(rl.max_kv_layers==limits.max_kv_layers && rl.max_kv_bytes==limits.max_kv_bytes,"legacy MTP limits unchanged");
        }
    }
#if defined(SUMMARY_DISK_TEST_GPU)
    cudaFree(ss.gdn_state);cudaFree(ss.ple_hist);
    fs::remove_all(dir);std::printf("summary_disk_gpu_test: %d checks passed (tiny real GPU transfers, no model)\n",checks);
#else
    fs::remove_all(dir);std::printf("summary_disk_cpu_test: %d checks passed (CPU transfers, no GPU/model)\n",checks);
#endif
}
