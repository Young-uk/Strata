// Process-local spill tier. Native session files retain the normal identity and
// checksum checks; only small prefix keys remain resident. No startup manifest.
#pragma once
#include "strata/core/conversation_file.hpp"
#include <filesystem>
#include <functional>
#include <cstdlib>
#include <cstdio>

namespace strata::core {
class ConversationDiskCache {
    struct Entry { SavedConversation keys; std::string path; size_t bytes; };
    std::vector<Entry> entries_;
    std::string directory_;
    std::string held_;
    size_t budget_ = 0, bytes_ = 0, serial_ = 0;
    void erase(size_t i) {
        std::error_code ec;
        std::filesystem::remove(entries_[i].path, ec);
        bytes_ -= entries_[i].bytes;
        entries_.erase(entries_.begin() + i);
    }
    static SavedConversation keys(const SavedConversation& image) {
        SavedConversation out;
        out.cvec = image.cvec;
        out.live.ids = image.live.ids; out.live.imgs = image.live.imgs;
        for (const auto& c : image.checkpoints) {
            ConversationCheckpoint k; k.ids = c.ids; k.imgs = c.imgs;
            out.checkpoints.push_back(std::move(k));
        }
        return out;
    }
public:
    ConversationDiskCache() = default;
    ConversationDiskCache(const ConversationDiskCache&) = delete;
    ConversationDiskCache& operator=(const ConversationDiskCache&) = delete;
    ~ConversationDiskCache() { close(); }
    void close() {
        if (!directory_.empty()) { std::error_code ec; std::filesystem::remove_all(directory_, ec); directory_.clear(); }
    }
    // The caller owns this private temporary directory and removes it on crash
    // recovery too. Never inspect or remove other processes' spill files.
    void configure(std::string directory, size_t budget) { directory_ = std::move(directory); budget_ = budget; }
    bool enabled() const { return !directory_.empty() && budget_ > 0; }
    void hold(std::string path) { held_ = std::move(path); }
    void unhold() { held_.clear(); }
    void clear() { while (!entries_.empty()) erase(0); }
    bool put(const SavedConversation& image, size_t estimate,
             const std::function<bool(const std::string&)>& write) {
        if (!enabled() || estimate == 0 || estimate > budget_) return false;
        auto key = keys(image);  // allocate keys before any publication
        while (!entries_.empty() && (entries_.size() >= 8 || bytes_ > budget_ - estimate)) {
            size_t i = 0;
            while (i < entries_.size() && entries_[i].path == held_) ++i;
            if (i == entries_.size()) return false;
            erase(i);
        }
        const std::string path = directory_ + "/" + std::to_string(++serial_) + ".session";
        struct Unpublished { std::string path; bool published = false;
            ~Unpublished() { if (!published) { std::error_code ec; std::filesystem::remove(path, ec); } }
        } pending{path};
        if (!write(path)) return false;
        std::error_code ec;
        const auto size = std::filesystem::file_size(path, ec);
        if (ec || size == 0 || size > estimate || size > budget_ - bytes_) return false;
        entries_.push_back({std::move(key), path, size});
        bytes_ += size; pending.published = true;
        return true;
    }
    template<class Token>
    ConversationCache::Match best(const std::vector<Token>& ids,
                                  const std::vector<ConversationImageKey>& imgs, bool cvec) const {
        ConversationCache::Match best;
        for (size_t i = entries_.size(); i-- > 0;) {
            const auto& e = entries_[i]; if (e.keys.cvec != cvec) continue;
            auto consider = [&](const auto& c, bool live) {
                const auto n = conversation_prefix(c, ids, imgs);
                if (n > best.tokens) best = {i, n, live, 0};
            };
            consider(e.keys.live, true);
            for (const auto& c : e.keys.checkpoints) consider(c, false);
        }
        return best;
    }
    const std::string& path(size_t i) const { return entries_.at(i).path; }
};
} // namespace strata::core
