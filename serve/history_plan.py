"""CPU-only retained-turn planning for the native four-token indexer contract."""
from __future__ import annotations
import struct


def token_hash(ids):
    value = 14695981039346656037
    for token in ids:
        for shift in (0, 8, 16, 24):
            value = ((value ^ ((token >> shift) & 255)) * 1099511628211) & ((1 << 64) - 1)
    return value


def restored_ids(path, binding):
    """Read only the bounded token table after native restore verified the file.

    Session v1/v2 have the same early geometry and token-table layout. Unknown
    formats stay usable for inference without making an alignment claim.
    """
    count = binding['tokens']
    if type(count) is not int or not 0 < count <= 262144:
        return None
    with open(path, 'rb') as source:
        header = source.read(64)
        if len(header) != 64 or struct.unpack_from('<II', header, 8) not in ((1, 64), (2, 64)):
            return None
        source.seek(64 + 18 * 8 + 3 * 8)
        size = source.read(8)
        if len(size) != 8 or struct.unpack('<Q', size)[0] != count:
            return None
        raw = source.read(count * 4)
        if len(raw) != count * 4:
            return None
    ids = list(struct.unpack('<' + str(count) + 'i', raw))
    return ids if all(token >= 0 for token in ids) and token_hash(ids) == binding['hash'] else None


def retained_match(old, new, turn):
    """Mirror include/history_cache.hpp without touching authoritative state."""
    common = 0
    while common < min(len(old), len(new)) and old[common] == new[common]:
        common += 1
    prefix = common
    while prefix and (prefix == len(new) or new[prefix] != turn):
        prefix -= 1
    if not prefix or prefix >= len(new) or new[prefix] != turn:
        return None
    turn_boundary = prefix
    prefix = (prefix + 3) // 4 * 4
    if prefix > common or len(new) - prefix <= 512:
        return None
    offset, best = prefix - turn_boundary, None
    for source in range(prefix + 4, len(old) - 511, 4):
        if old[source - offset] != turn or old[source:source + 32] != new[prefix:prefix + 32]:
            continue
        length = 32
        while source + length < len(old) and prefix + length < len(new) and old[source + length] == new[prefix + length]:
            length += 1
        length = min(length, len(new) - prefix - 1) // 4 * 4
        if length >= 512 and (best is None or length > best['retained_tokens']):
            best = {'prefix_tokens': prefix, 'source_tokens': source, 'retained_tokens': length}
    return best


def choose_boundary(snapshot, owner, candidates, encode, turn, context):
    """Prefer the requested boundary, then nearest earlier complete turns.

    The frontend owns semantic turn boundaries; this endpoint owns exact wire
    tokenization. Every candidate is budget-checked with summary/output reserve.
    Native admission still verifies the current cache token hash before FORK.
    """
    if not snapshot or snapshot.get('owner') != owner or not isinstance(snapshot.get('ids'), list):
        return {'boundary': None, 'reason': 'cache-unavailable'}
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 16:
        raise ValueError('history planning requires 1 to 16 candidates')
    previous = None
    for candidate in candidates:
        boundary = candidate.get('boundary') if isinstance(candidate, dict) else None
        if type(boundary) is not int or boundary <= 0 or (previous is not None and boundary >= previous):
            raise ValueError('history boundaries must be positive and strictly descending')
        previous = boundary
        ids = encode(candidate)
        # The native main prompt and the later handoff both need room. A plan
        # never adopts an overlarge retained envelope just to obtain cache reuse.
        if len(ids) + 4096 > context * 0.8:
            continue
        match = retained_match(snapshot['ids'], ids, turn)
        if match:
            return {'boundary': boundary, 'reason': 'aligned', **match}
    return {'boundary': None, 'reason': 'no-budgeted-aligned-turn'}
