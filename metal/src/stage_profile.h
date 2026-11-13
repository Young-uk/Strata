// Opt-in diagnostic for stage-boundary-only devices (M1). Splitting at existing
// full buffer barriers serializes profiling groups; these are not production timings.
#pragma once

#if !defined(STRATA_DS4_RUNTIME)
struct StageProfile {
    id<MTLCounterSampleBuffer> samples = nil;
    struct Group { std::string name; NSUInteger index; size_t dispatches; };
    std::vector<Group> groups;
    static std::string category(const Dispatch& d) {
        const auto& n = d.p->name;
        for (const char* prefix : {"mm_gu", "mm_down", "moe_gu_", "moe_down_", "gemv_", "gemm_f16", "qsa_attn", "qsa_score", "qsa_topk", "gdn_scan", "inc"})
            if (n.rfind(prefix, 0) == 0) return n;
        if (n.rfind("dq_f16", 0) == 0) return "weight dequant";
        return "other kernels";
    }
    StageProfile(id<MTLDevice> device, const std::vector<Dispatch>& ds, const std::vector<int>& cuts) {
        static const bool enabled = [] { const char* s = std::getenv("STRATA_METAL_STAGE_PROFILE"); return s && std::string(s) == "1"; }();
        static bool consumed = false;
        static size_t seen = 0;
        static const size_t at = [] {
            const char* e = std::getenv("STRATA_METAL_STAGE_PROFILE_AT");
            const long n = e ? std::strtol(e, nullptr, 10) : 1;
            return (size_t)std::max<long>(1, n);
        }();
        if (!enabled || consumed || ++seen < at) return;
        consumed = true; // one selected segmented graph, never every decode window
        if (![device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]) {
            std::fprintf(stderr, "metal stage profile: stage timestamps unavailable\n"); return;
        }
        id<MTLCounterSet> timestamps = nil;
        for (id<MTLCounterSet> set in device.counterSets)
            if ([set.name isEqualToString:MTLCommonCounterSetTimestamp]) timestamps = set;
        size_t expected = 0, begin = 0;
        for (size_t s = 0; s <= cuts.size(); ++s) {
            const size_t end = s < cuts.size() ? (size_t)cuts[s] : ds.size();
            while (begin < end) {
                const auto name = category(ds[begin]);
                size_t next = begin + 1;
                while (next < end && !(ds[next].barrier && category(ds[next]) != name)) ++next;
                ++expected; begin = next;
            }
        }
        // M1 permits at most 32 KiB of timestamp samples (2 x 8 bytes per group).
        if (!timestamps || expected > 2048) {
            std::fprintf(stderr, "metal stage profile: timestamp set unavailable or graph too large\n"); return;
        }
        MTLCounterSampleBufferDescriptor* desc = [MTLCounterSampleBufferDescriptor new];
        desc.counterSet = timestamps;
        desc.storageMode = MTLStorageModeShared;
        desc.sampleCount = std::max<size_t>(2, 2 * expected);
        NSError* error = nil;
        samples = [device newCounterSampleBufferWithDescriptor:desc error:&error];
        if (!samples) std::fprintf(stderr, "metal stage profile: %s\n", error.localizedDescription.UTF8String);
    }
    void segment(id<MTLCommandBuffer> cb, const std::vector<Dispatch>& ds, size_t first, size_t end) {
        while (first < end) {
            const std::string name = category(ds[first]);
            size_t next = first + 1;
            while (next < end && !(ds[next].barrier && category(ds[next]) != name)) ++next;
            const NSUInteger index = 2 * groups.size();
            MTLComputePassDescriptor* desc = [MTLComputePassDescriptor computePassDescriptor];
            desc.dispatchType = MTLDispatchTypeConcurrent;
            desc.sampleBufferAttachments[0].sampleBuffer = samples;
            desc.sampleBufferAttachments[0].startOfEncoderSampleIndex = index;
            desc.sampleBufferAttachments[0].endOfEncoderSampleIndex = index + 1;
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDescriptor:desc];
            for (size_t i = first; i < next; ++i) {
                if (i > first && ds[i].barrier) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                encode(enc, ds[i]);
                if ((i + 1) % 128 == 0) strata::core::metal_encoded(128);
            }
            [enc endEncoding];
            groups.push_back({name, index, next-first});
            first = next;
        }
    }
    void report() {
        if (!samples || groups.empty()) return;
        NSData* data = [samples resolveCounterRange:NSMakeRange(0, 2 * groups.size())];
        if (data.length != 2 * groups.size() * sizeof(MTLCounterResultTimestamp)) {
            std::fprintf(stderr, "metal stage profile: timestamp resolve failed\n"); return;
        }
        const auto* values = (const MTLCounterResultTimestamp*)data.bytes;
        std::map<std::string, std::pair<double, size_t>> totals;
        double sum = 0;
        size_t invalid = 0;
        for (const auto& g : groups) {
            const uint64_t a=values[g.index].timestamp, b=values[g.index+1].timestamp;
            if (a == MTLCounterErrorValue || b == MTLCounterErrorValue || b < a) { ++invalid; continue; }
            const double ms = (b-a)/1e6;
            auto& t = totals[g.name]; t.first += ms; t.second += g.dispatches; sum += ms;
        }
        std::vector<std::pair<double,std::string>> ranked;
        for (const auto& [name,t] : totals) ranked.push_back({t.first,name});
        std::sort(ranked.rbegin(),ranked.rend());
        std::fprintf(stderr,"metal stage profile: %zu encoder groups, %.3f ms GPU timestamps, %zu invalid; diagnostic split encoders, not normal-path throughput\n",groups.size(),sum,invalid);
        for(size_t i=0;i<std::min<size_t>(ranked.size(),30);++i) {
            const auto& [ms,name]=ranked[i];
            std::fprintf(stderr,"metal stage profile: %9.3f ms %5.1f%% %5zu dispatches %s\n",ms,sum?100*ms/sum:0,totals[name].second,name.c_str());
        }
    }
};
#endif
