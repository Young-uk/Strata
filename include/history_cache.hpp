#pragma once
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>
namespace strata::history {
struct Plan { size_t prefix = 0, source = 0, length = 0; };
inline bool rows_fit(size_t bytes, size_t stride, size_t prefix, size_t source, size_t length) {
    return stride && prefix < source && source <= bytes / stride && length <= bytes / stride - source;
}
inline bool compact_rows(void* data, size_t bytes, size_t stride, size_t prefix, size_t source, size_t length) {
    if (!data || !rows_fit(bytes, stride, prefix, source, length)) return false;
    std::memmove(static_cast<uint8_t*>(data) + prefix * stride,
                 static_cast<uint8_t*>(data) + source * stride, length * stride);
    return true;
}

// Only explicit historical handoffs may use this approximation. Four-token
// boundaries preserve the indexer's pooled keys without inventing raw keys.
template<class T> inline Plan match(const std::vector<int32_t>& old, const std::vector<T>& next, int32_t turn) {
    Plan best;
    size_t common = 0;
    while (common < old.size() && common < next.size() && old[common] == next[common]) ++common;
    size_t prefix = common;
    while (prefix && (prefix == next.size() || next[prefix] != turn)) --prefix;
    if (!prefix || prefix >= next.size() || next[prefix] != turn) return best;
    const size_t turn_boundary = prefix;
    prefix = (prefix + 3) / 4 * 4;
    // A shared turn header can extend the fixed prefix to the next pooled boundary.
    // Every extra prefix token must still be identical in the live and incoming histories.
    if (prefix > common || next.size() - prefix <= 512) return best;
    const size_t offset = prefix - turn_boundary;
    for (size_t src = prefix + 4; src + 512 <= old.size(); src += 4) {
        if (old[src-offset] != turn || !std::equal(next.begin() + prefix, next.begin() + prefix + 32, old.begin() + src)) continue;
        size_t n = 32;
        while (src + n < old.size() && prefix + n < next.size() && old[src+n] == next[prefix+n]) ++n;
        // Leave a fresh token before sampling and end on a whole pooled block.
        n = std::min(n, next.size() - prefix - 1) / 4 * 4;
        if (n >= 512 && n > best.length) best = {prefix, src, n};
    }
    return best;
}
inline bool positions_valid(const std::vector<int32_t>& positions, size_t tokens, int64_t limit = INT32_MAX) {
    if (positions.empty()) return true;
    if (positions.size() != tokens + 1) return false;
    int32_t last = -1;
    for (int32_t p : positions) { if (p <= last || p >= limit) return false; last = p; }
    return true;
}
} // namespace strata::history
