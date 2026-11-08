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
