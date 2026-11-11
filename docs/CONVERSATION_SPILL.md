# Experimental Metal conversation spill cache

Set `STRATA_CONVERSATION_DISK_DIR` to an existing private directory to enable
the process-local disk tier. The default path is unchanged when unset.
Memory parking remains first choice. Failed RAM parking streams authoritative
KV through the existing session-file writer, without a full host KV capture.
LRU evictions can write their already captured image. Files are selected by an
exact token/image/steering prefix and read through the normal model/config,
checksum, geometry and RAM admission checks. The restore source is held while
other cached branches are spilled and released to make room. If admission still
fails, normal prefill remains the fallback; no memory guard is bypassed.

The tier is bounded to 4 GiB and eight files, in a mode-0700 temporary child
directory. Normal QUIT explicitly removes it before the serving path's `_Exit`.
The next start removes only same-user spill directories whose process is gone.
Clean-shutdown `DiskSession` manifests remain independent and unchanged.

Short text-only clients can use `atomic_summary:true, think:false` for the
existing isolated lane. Set `atomic_auxiliary:true` as well when a short task
requires `response_format`. This does not enable tools, thinking, or permissions;
tool-result continuations belong on the ordinary conversation path. Ordinary
summary requests still reject structured output.

Host regression commands:

```sh
clang++ -std=c++20 -Iinclude src/core/conversation_disk_cache_test.cpp -o /tmp/conversation_disk_cache_test
/tmp/conversation_disk_cache_test
python -m unittest serve.test_atomic_auxiliary serve.test_atomic_summary
```

Actual model validation must include RAM parking refusal, LRU eviction, unchanged
deterministic output after restore, structured auxiliary work followed by prefix
reuse, cached summary cloning, and cleanup on normal termination. Test with a
separate directory and one engine at a time on memory-constrained Macs.
