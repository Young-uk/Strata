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

// Drop image ownership only if every image lies wholly outside retained rows.
// The main cached positions still carry the visual observations until moved.
template<class Image> inline bool images_discarded(const std::vector<int32_t>& ids,
        const std::vector<Image>& images, const Plan& plan, int32_t pad) {
    for (const auto& image : images) {
        if (image.start < 0 || uint64_t(image.start) >= ids.size() || ids[size_t(image.start)] != pad) return false;
        size_t end = size_t(image.start);
        while (end < ids.size() && ids[end] == pad) ++end;
        if (size_t(image.start) < plan.prefix ||
            (size_t(image.start) < plan.source + plan.length && end > plan.source)) return false;
    }
    return true;
}

// Only explicit historical handoffs may use this approximation. Four-token
// boundaries preserve the indexer's pooled keys without inventing raw keys.
template<class T> inline Plan match(const std::vector<int32_t>& old, const std::vector<T>& next, int32_t turn, int32_t image_pad = -1) {
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
    if (image_pad >= 0 && std::find(old.begin(), old.begin() + prefix, image_pad) != old.begin() + prefix) return best;
    const size_t offset = prefix - turn_boundary;
    for (size_t src = prefix + 4; src + 512 <= old.size(); src += 4) {
        if (old[src-offset] != turn || !std::equal(next.begin() + prefix, next.begin() + prefix + 32, old.begin() + src)) continue;
        size_t n = 32;
        while (src + n < old.size() && prefix + n < next.size() && old[src+n] == next[prefix+n]) ++n;
        // Reuse text only: visual rows and the following tail are freshly read.
        if (image_pad >= 0) {
            const auto image = std::find(old.begin() + src, old.begin() + src + n, image_pad);
            n = size_t(image - (old.begin() + src));
        }
        // Leave a fresh token before sampling and end on a whole pooled block.
        n = std::min(n, next.size() - prefix - 1) / 4 * 4;
        if (n >= 512 && n > best.length) best = {prefix, src, n};
    }
    return best;
}
// Retain authoritative cached positions and shift the fresh tail uniformly.
// Uniform translation preserves the incoming image grid's three-axis geometry.
inline bool compose_positions(const std::vector<int32_t>& old, const std::vector<int32_t>& incoming,
        size_t prefix, size_t source, size_t length, std::vector<int32_t>& out) {
    if (old.size() % 3 || incoming.size() % 3 || prefix > source || source >= old.size()/3 ||
        length >= old.size()/3 - source || prefix >= incoming.size()/3 || length >= incoming.size()/3 - prefix) return false;
    const size_t end = prefix + length;
    const int64_t shift = int64_t(old[(source + length)*3]) - incoming[end*3];
    auto next = incoming;
    for (size_t row = 0; row < prefix; ++row)
        for (size_t k = 0; k < 3; ++k) next[row*3+k] = old[row*3+k];
    for (size_t row = 0; row < length; ++row)
        for (size_t k = 0; k < 3; ++k) next[(prefix+row)*3+k] = old[(source+row)*3+k];
    for (size_t row = end; row < next.size()/3; ++row)
        for (size_t k = 0; k < 3; ++k) {
            const int64_t value = int64_t(incoming[row*3+k]) + shift;
            if (value < 0 || value > INT32_MAX) return false;
            next[row*3+k] = int32_t(value);
        }
    out.swap(next);
    return true;
}
inline bool positions_valid(const std::vector<int32_t>& positions, size_t tokens, int64_t limit = INT32_MAX) {
    if (positions.empty()) return true;
    if (positions.size() != tokens + 1) return false;
    int32_t last = -1;
    for (int32_t p : positions) { if (p <= last || p >= limit) return false; last = p; }
    return true;
}
// Format v3 retains all three rotary axes for image-bearing approximate history.
// Repeated time positions are permitted only inside an owned, complete image grid.
template<class Image> inline bool position_map_valid(const std::vector<int32_t>& positions,
        const std::vector<int32_t>& ids, const std::vector<Image>& images, int64_t limit = INT32_MAX) {
    if (positions.empty()) return true;
    if (positions.size() == ids.size()+1) return images.empty() && positions_valid(positions,ids.size(),limit);
    if (images.empty() || ids.size() >= SIZE_MAX/3 || positions.size() != 3*(ids.size()+1)) return false;
    size_t next_image=0, row=0;
    int64_t last=-1;
    while (row <= ids.size()) {
        for(size_t k=0;k<3;k++) if(positions[row*3+k]<0 || positions[row*3+k]>=limit) return false;
        if(next_image<images.size() && images[next_image].start<0) return false;
        const size_t start=next_image<images.size()?size_t(images[next_image].start):ids.size()+1;
        if(start<row || (next_image<images.size() && start>=ids.size())) return false;
        const int64_t base=positions[row*3];
        if(base<=last) return false;
        if(row!=start) {
            if(positions[row*3+1]!=base || positions[row*3+2]!=base) return false;
            last=base;row++;continue;
        }
        const int32_t pad=ids[row];size_t end=row+1;
        const size_t following=next_image+1<images.size()?size_t(images[next_image+1].start):ids.size();
        if(following<=start || following>ids.size()) return false;
        while(end<following && ids[end]==pad) end++;
        int64_t ph=base,pw=base,width=0;
        for(size_t j=row;j<end;j++) {
            const int64_t t=positions[j*3],h=positions[j*3+1],w=positions[j*3+2];
            if(t!=base || h<base || w<base || h>=limit || w>=limit) return false;
            if(j==row) { if(h!=base || w!=base) return false; }
            else if(h==ph) { if(w!=pw+1) return false; }
            else {
                if(h!=ph+1 || w!=base) return false;
                const int64_t observed=pw-base+1;
                if(width && width!=observed) return false;
                width=observed;
            }
            last=std::max(last,std::max(h,w));ph=h;pw=w;
        }
        if(width && ((end-row)%size_t(width)!=0 || pw-base+1!=width)) return false;
        row=end;next_image++;
    }
    return next_image==images.size();
}
} // namespace strata::history
