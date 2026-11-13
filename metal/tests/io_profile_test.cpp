#include "../src/engine/io_profile.h"
#include <cassert>
#include <string>
int main() {
    using namespace strata::metal_engine;
    IoCounters a{100,2,1000,10,4,2,2.0,4.0}, b{350,5,51000,15,7,4,7.0,12.0};
    auto d=io_delta(a,b);
    assert(d.bytes==250 && d.experts==3 && d.worker_us==50000);
    assert(d.hits==5 && d.pre_hits==3 && d.misses==2 && d.stall_ms==5 && d.callback_ms==8);
    IoProfile p;p.add(d,4096,20.0);assert(p.total.worker_us/1000.0>p.gpu_span_ms);
    FILE* f=tmpfile();assert(f);p.report(f,false,true);rewind(f);char text[1024]={};assert(fread(text,1,sizeof(text)-1,f)>0);fclose(f);
    assert(std::string(text).find("expert_wait_ms=5.000")!=std::string::npos);
    assert(std::string(text).find("worker time overlaps")!=std::string::npos);
    auto reset=io_delta(b,a);assert(reset.bytes==0 && reset.stall_ms==0);
}
