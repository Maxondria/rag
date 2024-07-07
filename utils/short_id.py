import hashlib


def generate_short_id(content: str) -> str:
    hashing_algorithm = hashlib.sha256()
    hashing_algorithm.update(content.encode("utf-8"))
    return hashing_algorithm.hexdigest()
