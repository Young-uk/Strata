#include "strata/core/summary_transaction.hpp"
#include "strata/core/conversation_cache.hpp"
#include "strata/prefill/kva_floor.hpp"
#include <cstdio>
#include <cstdlib>
#include <type_traits>

using namespace strata::core;
static int checks = 0;
static void check(bool ok, const char* what) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); std::exit(1); }
}
int main() {
    static_assert(std::is_nothrow_move_constructible<ConversationCache>::value, "hold must not allocate");
    static_assert(std::is_nothrow_move_assignable<ConversationCache>::value, "release must not allocate");
    static_assert(std::is_nothrow_move_constructible<strata::prefill::KvaFloorRegistry>::value, "hold must not allocate");
    SummaryLimits limits;
    std::string args = "4096 1073741824 2147483648 /private/main snapshot";
    check(limits.parse(args) && args == "/private/main snapshot", "v2 parses explicit envelope and spaced path");
    check(limits.covers(4096, 1073741824, 2147483648), "qualified envelope inclusive boundary");
    check(!limits.covers(4097, 1, 1), "unqualified main token length refused");
    check(!limits.covers(1, 1073741825, 1), "unqualified snapshot refused");
    check(!limits.covers(1, 1, 2147483649), "unqualified protected cache footprint refused");
    for (const char* bad : {"/old/v1/path", "0 1 1 /path", "1 -1 1 /path", "1 1 1 ",
                            "1 1x 1 /path", "1 1 18446744073709551616 /path"}) {
        std::string invalid = bad; SummaryLimits other;
        check(!other.parse(invalid), "malformed/unbounded v2 envelope refused");
    }
    const uint64_t peak = uint64_t(2) << 30, floor = uint64_t(1) << 30;
    check(SummaryLimits::admits(peak + SummaryLimits::scratch + floor, peak, floor), "rollback peak inclusive admission");
    check(!SummaryLimits::admits(peak + SummaryLimits::scratch + floor - 1, peak, floor), "SAVE-sized but rollback-insufficient RAM refused");
    check(!SummaryLimits::admits(std::nullopt, peak, floor), "unknown telemetry refuses BEGIN");
    check(!SummaryLimits::admits(UINT64_MAX, UINT64_MAX, 0), "overflow refuses BEGIN");
    check(!SummaryLimits::admits(floor - 1, 0, floor), "floor underflow refused");
    check(SummaryTransaction::empty_eligible(true, false, 0, 0, 0, 0, 0, false), "explicit pristine empty main admitted");
    check(!SummaryTransaction::empty_eligible(true, true, 0, 0, 0, 0, 0, false), "valid live marker prevents empty admission");
    check(!SummaryTransaction::empty_eligible(true, false, 1, 0, 0, 0, 0, false), "stale live tokens prevent empty admission");
    check(!SummaryTransaction::empty_eligible(false, false, 0, 0, 0, 0, 0, false), "cancelled first prefill is not pristine even when buffers empty");
    check(!SummaryTransaction::empty_eligible(true, false, 0, 1, 0, 0, 0, false), "checkpoint without valid live is not empty");
    check(!SummaryTransaction::empty_eligible(true, false, 0, 0, 1, 0, 0, false), "parked cache prevents unqualified empty admission");
    check(!SummaryTransaction::empty_eligible(true, false, 0, 0, 0, 1, 0, false), "retained cache prevents empty admission");
    check(!SummaryTransaction::empty_eligible(true, false, 0, 0, 0, 0, 1, false), "vision ownership prevents empty admission");
    check(!SummaryTransaction::empty_eligible(true, false, 0, 0, 0, 0, 0, true), "poisoned empty state refused");
    for (const char* command : {"GEN bad", "GENI bad", "BGEN bad", "BGENI bad", "RESTORE missing"})
        check(SummaryTransaction::invalidates_pristine(command), "ordinary attempt invalidates pristine before parsing or prefill");
    check(!SummaryTransaction::invalidates_pristine("SUMMARY_BEGIN 1 2 3 /path"), "summary control keeps original provenance until hold");
    check(SummaryLimits::shape_admits(0, false, false, 0, 0, 0), "cold text shape allowed");
    check(SummaryLimits::shape_admits(0, false, false, 1, 0, 0), "one foreground checkpoint allowed");
    check(!SummaryLimits::shape_admits(1, false, false, 1, 0, 0), "image markers refused");
    check(!SummaryLimits::shape_admits(0, true, false, 1, 0, 0), "embedding ownership refused");
    check(!SummaryLimits::shape_admits(0, false, true, 1, 0, 0), "vision position state refused");
    check(SummaryLimits::shape_admits(0, false, false, 6, 4, 4), "six checkpoints and four configured parked slots allowed");
    check(!SummaryLimits::shape_admits(0, false, false, 7, 0, 4), "seventh checkpoint refused");
    check(!SummaryLimits::shape_admits(0, false, false, 1, 1, 0), "parked count above configured slots refused");
    check(!SummaryLimits::shape_admits(0, false, false, 1, 5, 5), "fifth parked entry refused despite configured slots");
    const SummaryHostHealth healthy{1, 1024, 99};
    check(SummaryLimits::health_admits(healthy, healthy), "stable existing swap allowed");
    check(SummaryLimits::health_admits(healthy, SummaryHostHealth{1, 512, 99}), "falling swap allowed");
    check(!SummaryLimits::health_admits({}, healthy), "unknown baseline refused");
    check(!SummaryLimits::health_admits(healthy, {}), "unknown current health refused");
    for (uint32_t pressure : {0u, 2u, 4u, 8u}) {
        check(!SummaryLimits::health_admits(healthy, SummaryHostHealth{pressure, 1024, 99}), "non-normal current pressure refused");
        check(!SummaryLimits::health_admits(SummaryHostHealth{pressure, 1024, 99}, healthy), "non-normal baseline pressure refused");
    }
    check(!SummaryLimits::health_admits(healthy, SummaryHostHealth{1, 1025, 99}), "growing swap refused");
    check(!SummaryLimits::health_admits(healthy, SummaryHostHealth{1, 512, 100}), "swapout churn refused even when usage falls");
    check(!SummaryLimits::health_admits(healthy, SummaryHostHealth{1, 512, 98}), "counter regression fails closed");
    {
        auto image = [] {
            SavedConversation e; e.live.ids = {10, 11, 12};
            e.checkpoints.resize(6);
            for (auto& c : e.checkpoints) { c.ids = {10}; c.gdn.resize(16, 7); }
            return e;
        };
        auto admit_image = [&](SavedConversation e) {
            ConversationCache c(1 << 20, 4); check(c.put(std::move(e)), "fixture image put");
            return c.summary_text_shape(2048, 6, 4);
        };
        check(admit_image(image()), "six parked text checkpoints qualified");
        auto seven = image(); seven.checkpoints.resize(7);
        check(!admit_image(std::move(seven)), "seven parked checkpoints refused");
        auto deep = image(); deep.checkpoints.back().imgs.push_back({0, 33});
        check(!admit_image(std::move(deep)), "deep parked checkpoint image refused");
        auto live_image = image(); live_image.live.imgs.push_back({0, 33});
        check(!admit_image(std::move(live_image)), "parked live image refused");
        auto stage = image(); stage.checkpoints.back().stage_parts.resize(1);
        check(!admit_image(std::move(stage)), "nested checkpoint stage refused");
        auto staged_image = image(); staged_image.stage_images.resize(1);
        check(!admit_image(std::move(staged_image)), "parked stage image refused");
        auto long_image = image(); long_image.live.ids.resize(2049);
        check(!admit_image(std::move(long_image)), "long parked live extent refused");
        auto long_check = image(); long_check.checkpoints.back().ids.resize(2049);
        check(!admit_image(std::move(long_check)), "long parked checkpoint extent refused");
        ConversationCheckpoint c; c.ids = {1};
        check(c.summary_text_shape(2048), "text main checkpoint qualified");
        c.imgs.push_back({0, 9}); check(!c.summary_text_shape(2048), "image in main checkpoint refused");
        ConversationCache reuse(1 << 20, 4);
        auto kv = [] { std::vector<ConversationKv> v(1); v[0].k.resize(64, 3); return v; };
        reuse.retain(kv(), 2, {}, {1, 2});
        check(!reuse.summary_text_shape(2048, 6, 4), "unknown reuse provenance refused");
        reuse.retain(kv(), 2, {}, {1, 2}, true);
        check(reuse.summary_text_shape(2048, 6, 4), "known bounded text reuse admitted");
        reuse.retain(kv(), 2049, {}, std::vector<int32_t>(2049), true); reuse.limit_reuse(1);
        check(!reuse.summary_text_shape(2048, 6, 4), "captured reuse extent bounded even after prefix shrinks");
        reuse.retain(kv(), 2, {}, {1}, true);
        check(!reuse.summary_text_shape(2048, 6, 4), "reuse identity extent mismatch refused");
        std::vector<std::vector<ConversationKv>> stages; stages.push_back(kv());
        reuse.retain(kv(), 2, std::move(stages), {1, 2}, true);
        check(!reuse.summary_text_shape(2048, 6, 4), "staged reuse refused");
        SummaryLimits bounded; bounded.tokens = 2048; bounded.snapshot = 256ull << 20; bounded.protected_bytes = 1ull << 30;
        check(bounded.covers(2048, 256ull << 20, 1ull << 30), "candidate byte boundary inclusive");
        check(!bounded.covers(2048, 256ull << 20, (1ull << 30) + 1), "total protected excess refused even for legal shape");
    }
    SummaryTransaction cold;
    for (int batch = 0; batch < 2; ++batch) {
        cold.commit_begin("/cold", true);
        check(cold.empty_mode() && !cold.can_end("/wrong"), "empty mode keeps exact path identity");
        cold.begin_restore();
        check(cold.empty_mode() && !cold.allows("GEN 1 1,2"), "failed empty END seals inference");
        cold.commit_end();
        check(!cold.active() && !cold.empty_mode(), "successful empty release allows repeated initial batches");
    }
    SummaryTransaction tx;
    check(!tx.can_begin("") && tx.can_begin("/private/main"), "empty snapshot path rejected");
    check(!tx.can_end("/private/main"), "END before BEGIN rejected");
    // A failed disk write never commits BEGIN.
    check(!tx.active() && tx.allows("GEN 1 1,2"), "failed save leaves foreground usable");
    tx.commit_begin("/private/main");
    check(!tx.can_begin("/private/main"), "duplicate BEGIN rejected");
    check(!tx.can_end("/private/other") && tx.can_end("/private/main"), "exact restore path required");
    for (const char* command : {"SAVE /other", "RESTORE /other", "SUMMARY_BEGIN /other", "GENI 1 img 1,2", "BGEN 0 1 1,2", "VRAM", "RESET"})
        check(!tx.allows(command), "mutating commands sealed during hold");
    check(tx.allows("GEN 1 1,2") && tx.allows("SUMMARY_CAPS"), "bounded text admission and capability probe allowed");
    check(SummaryTransaction::bounded(16375, 1), "inclusive safe16K bound");
    check(!SummaryTransaction::bounded(16376, 1), "over budget rejected");
    check(!SummaryTransaction::bounded(2, INT64_MAX), "overflow rejected");
    check(!SummaryTransaction::bounded(2, 0), "prefill-only rejected");
    tx.begin_restore();
    check(!tx.allows("GEN 1 1,2") && tx.active(), "failed END remains sealed against inference");
    check(tx.can_end("/private/main"), "failed END may retry exact snapshot");

    ConversationCache cache(4096, 3), held(0, 0);
    for (int token : {11, 22}) {
        SavedConversation image;
        image.live.ids = {token, token}; image.live.gdn.resize(64, token);
        check(cache.put(std::move(image)), "populate parked cache");
    }
    cache.count_borrow();
    auto match = cache.best(std::vector<int32_t>{22, 22, 33}, {}, true);
    const auto* original_entry = cache.find(match.serial);
    const auto* original_bytes = original_entry->live.gdn.data();
    const auto bytes = cache.bytes();
    std::vector<ConversationCheckpoint> chain(6), protected_chain;
    for (size_t i = 0; i < chain.size(); ++i) { chain[i].ids = {int32_t(i)}; chain[i].used = 100 + i; }
    auto* original_chain = chain.data();
    chain.swap(protected_chain); std::swap(cache, held);
    check(chain.empty() && !cache.enabled(), "summary cache starts empty and disabled");
    cache.clear(); // summary non-finite recovery cannot touch protected state
    chain.swap(protected_chain); std::swap(cache, held);
    check(chain.data() == original_chain && chain.size() == 6 && chain[5].used == 105, "full checkpoint chain moved exactly");
    check(cache.find(match.serial) == original_entry && original_entry->live.gdn.data() == original_bytes,
          "parked identity and buffers preserved without copies");
    check(cache.bytes() == bytes && cache.borrowed() == 1, "cache counters preserved");
    check(cache.evict_oldest() && cache.find(match.serial) == original_entry, "original LRU order preserved");
    // Vision's main embeddings must retain addresses while text resets M-RoPE to identity.
    std::vector<float> rows{1.25f, 2.5f}, held_rows;
    std::vector<const float*> pointers{rows.data()}, held_pointers;
    std::vector<int32_t> positions{0, 4, 8}, held_positions = positions;
    const float* original_rows = rows.data();
    rows.swap(held_rows); pointers.swap(held_pointers);
    positions = {0, 0, 0}; // summary text position table
    positions.swap(held_positions); rows.swap(held_rows); pointers.swap(held_pointers);
    check(rows.data() == original_rows && pointers[0] == rows.data() && *pointers[0] == 1.25f,
          "vision embedding pointer ownership restored exactly");
    check(positions == std::vector<int32_t>({0, 4, 8}), "vision M-RoPE host table restored");
    tx.commit_end();
    check(!tx.active() && tx.allows("GENI 1 img 1,2"), "successful restore releases foreground");
    check(!tx.can_end("/private/main"), "duplicate END rejected");
    std::printf("summary_transaction_test: %d checks passed\n", checks);
}
