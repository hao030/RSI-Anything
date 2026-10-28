import hashlib

def stable_seed(*parts):
    payload = ':'.join((str(part) for part in parts))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:4], 'little')
