#include "strata/core/conversation_disk_cache.hpp"
#include <cassert>
#include <fstream>
#include <unistd.h>
using namespace strata::core;
int main() {
    char temp[] = "/tmp/strata-spill-test-XXXXXX";
    const std::string dir = mkdtemp(temp);
    {
        ConversationDiskCache cache;
        cache.configure(dir, 100);
        SavedConversation a; a.cvec = false; a.live.ids = {1,2,3}; a.live.imgs = {{1, 42}};
        auto writer = [](const std::string& p) { std::ofstream(p) << std::string(60, 'x'); return true; };
        assert(cache.put(a, 60, writer));
        std::vector<int> prompt{1,2,3,4};
        assert(cache.best(prompt, {{1,42}}, false).tokens == 3);
        assert(cache.best(prompt, {{1,43}}, false).tokens == 0);
        assert(cache.best(prompt, {{1,42}}, true).tokens == 0);
        assert(!cache.put(a, 101, writer));
        assert(cache.best(prompt, {{1,42}}, false).tokens == 3);
        SavedConversation b; b.cvec = false; b.live.ids = {5,6,7};
        cache.hold(cache.path(0));
        assert(!cache.put(b, 60, writer));
        assert(cache.best(prompt, {{1,42}}, false).tokens == 3);
        cache.unhold();
        assert(cache.put(b, 60, writer));
        assert(cache.best(prompt, {{1,42}}, false).tokens == 0);
        assert(!cache.put(a, 40, [](const auto& p) { std::ofstream(p) << "partial"; return false; }));
        assert(cache.best(std::vector<int>{5,6,7,8}, {}, false).tokens == 3);
        assert(std::distance(std::filesystem::directory_iterator(dir), std::filesystem::directory_iterator{}) == 1);
        ConversationCache ram(1000, 1);
        bool spilled = false, cleared = false;
        ram.before_evict = [&](const auto& image) { spilled = image.live.ids == b.live.ids; };
        ram.after_clear = [&] { cleared = true; cache.clear(); };
        SavedConversation entry; entry.live.ids = b.live.ids;
        assert(ram.put(std::move(entry))); assert(ram.evict_oldest()); assert(spilled);
        ram.clear(); assert(cleared);
        assert(cache.best(std::vector<int>{5,6,7,8}, {}, false).tokens == 0);
    }
    assert(!std::filesystem::exists(dir));
    std::puts("conversation spill identity, budget, failed publication, eviction and cleanup: passed");
}
