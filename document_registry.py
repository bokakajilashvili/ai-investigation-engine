
import hashlib
import json
from pathlib import Path

REGISTRY_PATH = Path("document_registry.json")


def calculate_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def load_registry() -> dict:
    if not REGISTRY_PATH.exists():
        return {}

    return json.loads(
        REGISTRY_PATH.read_text(encoding="utf-8")
    )


def find_duplicate(file_hash: str):
    registry = load_registry()

    for file_id, document in registry.items():
        if document["hash"] == file_hash:
            return {
                "file_id": file_id,
                "filename": document["filename"]
            }

    return None


def register_document(
    file_id: str,
    filename: str,
    file_hash: str
):
    registry = load_registry()

    registry[file_id] = {
        "filename": filename,
        "hash": file_hash
    }

    REGISTRY_PATH.write_text(
        json.dumps(registry, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

def remove_document(file_id: str):
    registry = load_registry()

    if file_id not in registry:
        return False

    del registry[file_id]

    REGISTRY_PATH.write_text(
        json.dumps(registry, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    

    return True
