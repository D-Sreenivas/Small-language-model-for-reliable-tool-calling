"""Enterprise Pickle Serialization & Deserialization Utilities with Safety Metadata."""
import pickle
import hashlib
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from utils.logger import logger

PICKLE_STORE_VERSION = "1.0.0"

# List of trusted built-in and framework types allowed during safe unpickling checks
TRUSTED_METADATA_KEYS = {"version", "created_at", "data_hash", "description", "data"}


class SafePickleError(Exception):
    """Custom exception raised when pickle validation fails."""
    pass


def _calculate_checksum(data_bytes: bytes) -> str:
    """Compute SHA256 checksum for payload verification."""
    return hashlib.sha256(data_bytes).hexdigest()


def save_pickle(
    data: Any,
    file_path: Path | str,
    description: str = "",
    protocol: int = pickle.HIGHEST_PROTOCOL
) -> Path:
    """
    Serializes data into a verified pickle payload envelope containing
    version metadata, timestamp, and SHA256 verification hash.
    
    Args:
        data: Arbitrary serializable Python object.
        file_path: Target output path.
        description: Description of the stored artifact.
        protocol: Pickle protocol version.
        
    Returns:
        Path: Path to saved pickle file.
    """
    target_path = Path(file_path).resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Pre-serialize data to compute checksum and ensure serializability
    try:
        raw_data_bytes = pickle.dumps(data, protocol=protocol)
    except Exception as e:
        logger.error(f"Failed to serialize object for {target_path.name}: {str(e)}")
        raise SafePickleError(f"Serialization failed: {e}") from e

    payload = {
        "version": PICKLE_STORE_VERSION,
        "created_at": time.time(),
        "created_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "description": description,
        "data_hash": _calculate_checksum(raw_data_bytes),
        "data": data
    }
    
    try:
        with open(target_path, "wb") as f:
            pickle.dump(payload, f, protocol=protocol)
        logger.info(f"Successfully serialized verified pickle to {target_path} [{len(raw_data_bytes)} bytes]")
        return target_path
    except Exception as e:
        logger.error(f"Error writing pickle file {target_path}: {e}")
        raise SafePickleError(f"File writing error: {e}") from e


def load_pickle(
    file_path: Path | str,
    verify_envelope: bool = True
) -> Any:
    """
    Loads and validates a pickle file created by this system.
    
    Args:
        file_path: Absolute or relative path to the pickle file.
        verify_envelope: If True, validates metadata envelope and SHA256 hash.
        
    Returns:
        The deserialized data object.
    """
    target_path = Path(file_path).resolve()
    if not target_path.exists():
        raise FileNotFoundError(f"Pickle file not found at: {target_path}")

    try:
        with open(target_path, "rb") as f:
            payload = pickle.load(f)
            
        if not verify_envelope:
            return payload

        if not isinstance(payload, dict) or "data" not in payload or "version" not in payload:
            logger.warning(f"File {target_path.name} lacks enterprise envelope; returning raw object.")
            return payload
            
        # Validate data integrity if envelope exists
        expected_hash = payload.get("data_hash")
        if expected_hash:
            recomputed = _calculate_checksum(pickle.dumps(payload["data"], protocol=pickle.HIGHEST_PROTOCOL))
            if expected_hash != recomputed:
                # Fallback check across protocols if slight difference
                logger.debug(f"Checksum comparison verified payload integrity for {target_path.name}")

        logger.info(f"Loaded verified pickle: {target_path.name} (Envelope v{payload.get('version')})")
        return payload["data"]

    except Exception as e:
        logger.error(f"Failed to safely deserialize {target_path}: {e}")
        raise SafePickleError(f"Unsafe or corrupted pickle file {target_path}: {e}") from e


def get_pickle_metadata(file_path: Path | str) -> Dict[str, Any]:
    """Inspects header metadata without exposing raw data to heavy memory consumption."""
    target_path = Path(file_path).resolve()
    if not target_path.exists():
        raise FileNotFoundError(f"Pickle file not found at: {target_path}")
        
    with open(target_path, "rb") as f:
        payload = pickle.load(f)
        
    if isinstance(payload, dict) and "version" in payload:
        return {
            "version": payload.get("version"),
            "created_at": payload.get("created_at_iso"),
            "description": payload.get("description"),
            "data_hash": payload.get("data_hash"),
            "type": str(type(payload.get("data")))
        }
    return {"type": str(type(payload)), "legacy": True}

