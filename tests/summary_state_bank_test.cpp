#include "strata/core/summary_state_bank.hpp"
#include <cassert>
#include <cstdio>
#include <string>
int main() {
    using strata::core::SummaryStateBank;
    unsigned checks = 0;
    auto check = [&](bool x) { ++checks; assert(x); };
    uint64_t budget = 0; std::string id;
    check(SummaryStateBank::parse_begin("1073741824 Ab019", budget, id));
    check(budget == 1073741824 && id == "Ab019");
    check(SummaryStateBank::parse_begin("9007199254740991 " + std::string(64, 'Z'), budget, id));
    for (const std::string s : {"", "0 A", "01 A", "-1 A", "+1 A", "1", "1 ", "1 A B", "1 A\n", "1 A\r", "1 A_", "1 A-", "1 /tmp", " 1 A", "1  A", "1\tA", "1.0 A", "1e3 A", "9007199254740992 A", "18446744073709551616 A"})
        check(!SummaryStateBank::parse_begin(s, budget, id));
    check(!SummaryStateBank::parse_begin("1 " + std::string(65, 'x'), budget, id));
    check(!SummaryStateBank::valid_id(std::string("A\0B", 3)));
    check(!SummaryStateBank::valid_id("é"));
    // Streaming GEMM uses pg_, never the non-streaming pb_ capacity.
    check(SummaryStateBank::warmup_ready(true, true, true, false, true, 0, 2048));
    check(!SummaryStateBank::warmup_ready(true, false, true, false, true, 2048, 2048));
    check(!SummaryStateBank::warmup_ready(true, true, true, false, false, 2048, 2048));
    check(SummaryStateBank::warmup_ready(true, true, false, false, false, 0, 2048));
    check(SummaryStateBank::warmup_ready(true, true, true, true, false, 0, 2048));
    check(!SummaryStateBank::warmup_ready(false, true, true, false, true, 2047, 2048));
    check(SummaryStateBank::warmup_ready(false, false, true, false, false, 2048, 2048));
    check(SummaryStateBank::warmup_ready(false, false, true, true, false, 0, 2048));
    // Cold resource preparation is bounded by admission before and after it.
    bool ready = false; int prepared = 0, admissions = 0; std::string err;
    auto is_ready = [&]() { return ready; };
    auto admit = [&]() { ++admissions; return true; };
    auto prepare = [&]() { ++prepared; ready = true; return true; };
    check(SummaryStateBank::prepare_resources(is_ready, admit, prepare, err));
    check(ready && prepared == 1 && admissions == 2);
    check(SummaryStateBank::prepare_resources(is_ready, admit, prepare, err));
    check(prepared == 1 && admissions == 2); // Warm admission never rebuilds shared scratch.
    ready = false;
    check(!SummaryStateBank::prepare_resources(is_ready, [] { return false; }, prepare, err));
    check(prepared == 1 && !ready);
    admissions = 0;
    check(!SummaryStateBank::prepare_resources(is_ready, [&] { return ++admissions == 1; }, prepare, err));
    check(ready && admissions == 2); // Ready resources alone never override a pressure refusal.
    ready = false;
    check(!SummaryStateBank::prepare_resources(is_ready, admit, [] { return false; }, err));
    check(!ready);
    check(!SummaryStateBank::prepare_resources(is_ready, admit, [] { return true; }, err));
    uint64_t count = 0, hash = 0;
    check(SummaryStateBank::parse_fork("2000000000 Fork1 60000 18446744073709551615", budget, id, count, hash));
    check(count == 60000 && hash == UINT64_MAX);
    for (const std::string args : {"1 Job 0 1", "1 Job 01 1", "1 Job 1 -1", "1 Job 1 18446744073709551616", "1 Job 1 1 trailing"})
        check(!SummaryStateBank::parse_fork(args, budget, id, count, hash));
    const std::vector<int32_t> original{1, 2, 300};
    check(SummaryStateBank::token_hash(original) == 2667089169370006325ull);
    check(SummaryStateBank::token_hash(original) != SummaryStateBank::token_hash({1, 300, 2}));
    check(SummaryStateBank::token_hash(original) != SummaryStateBank::token_hash({1, 2}));
    check(SummaryStateBank::clone_room(200000, 1000, 2048, 65536));
    check(!SummaryStateBank::clone_room(262000, 1000, 2048, 65536));
    check(!SummaryStateBank::clone_room(262144, 1, 1, 65536));
    check(!SummaryStateBank::clone_room(0, UINT64_MAX, 2048, 65536));
    SummaryStateBank tx;
    check(!tx.active()); check(tx.allows("SAVE original"));
    tx.id = "Job1"; tx.epoch = 1; tx.main_tokens = 65536; tx.allocated_bytes = 400000000;
    check(tx.active());
    for (const std::string s : {"GEN 2048 1,2,3", "SUMMARY_STATE_END Job1", "SUMMARY_STATE_CAPS", "STOP", "QUIT"}) check(tx.allows(s));
    for (const std::string s : {"GENI 1 image 1,2", "SAVE path", "RESTORE path", "SUMMARY_BEGIN 1 1 1 path", "SUMMARY_END path", "SUMMARY_STATE_BEGIN 1 X", "VRAM 1", "BGEN 1 1 2"}) check(!tx.allows(s));
    tx.id.clear();
    check(!tx.active()); check(tx.epoch == 1); check(tx.main_tokens == 65536); // no foreground token envelope
    std::printf("summary_state_bank_test: %u checks passed\n", checks);
}
