"""Opt-in prompt-prefix diagnostics. Hashes only; never used for cache admission."""
from __future__ import annotations
import hashlib
import array
import sys
import json
import re

MAX_MESSAGES = 2048
_HEX = re.compile(r'[a-f0-9]{64}')


def digest(value):
    h = hashlib.sha256()
    for part in json.JSONEncoder(sort_keys=True, separators=(',', ':'), ensure_ascii=True).iterencode(value):
        h.update(part.encode('ascii'))
    return h.hexdigest()


def token_digest(ids):
    packed = array.array('I', ids)
    if packed.itemsize != 4:
        raise ValueError('unsupported trace token width')
    if sys.byteorder != 'little':
        packed.byteswap()
    return hashlib.sha256(packed).hexdigest()


def capture(messages, tools, kwargs, ids, owner):
    if len(messages) > MAX_MESSAGES or not isinstance(owner, str) or not owner:
        return None
    return {'version': 1, 'owner_hash': digest(owner), 'tools': digest(tools), 'template': digest(kwargs),
            'messages': [digest(message) for message in messages], 'input_tokens': len(ids),
            'input_hash': token_digest(ids)}


def validated(value):
    if not isinstance(value, dict) or set(value) != {'version', 'owner_hash', 'tools', 'template', 'messages', 'input_tokens', 'input_hash'}:
        return None
    hashes = [value.get(key) for key in ('owner_hash', 'tools', 'template', 'input_hash')]
    messages = value.get('messages')
    if (type(value.get('version')) is not int or value['version'] != 1 or type(value.get('input_tokens')) is not int or
            not 0 <= value['input_tokens'] <= 2**31 - 1 or not isinstance(messages, list) or
            len(messages) > MAX_MESSAGES or any(not isinstance(x, str) or not _HEX.fullmatch(x)
                                               for x in [*hashes, *messages])):
        return None
    return {**value, 'messages': list(messages)}


def compare(previous, current, ids):
    current, previous = validated(current), validated(previous)
    if current is None:
        return None
    if previous is None:
        return {'changes': ['no_previous_trace'], 'input_tokens': len(ids)}
    changes = []
    for key in ('tools', 'template'):
        if previous[key] != current[key]:
            changes.append(key + '_changed')
    common = 0
    for a, b in zip(previous['messages'], current['messages']):
        if a != b:
            break
        common += 1
    if common < min(len(previous['messages']), len(current['messages'])):
        changes.append('historical_message_changed')
    elif len(current['messages']) < len(previous['messages']):
        changes.append('history_shortened')
    elif len(current['messages']) > len(previous['messages']):
        changes.append('history_appended')
    n = previous['input_tokens']
    prefix_equal = n <= len(ids) and token_digest(ids[:n]) == previous['input_hash']
    if not prefix_equal:
        changes.append('previous_rendered_input_prefix_changed')
    if not changes:
        changes.append('same_request_input')
    return {'changes': changes, 'common_input_messages': common,
            'first_changed_message': common if 'historical_message_changed' in changes else None,
            'previous_input_tokens': n, 'input_tokens': len(ids), 'previous_input_prefix_equal': prefix_equal}
