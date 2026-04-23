import threading
from typing import Dict

# Global dicts: workspace_dir_name -> accumulated tokens
_output_token_counts: Dict[str, int] = {}
_input_token_counts: Dict[str, int] = {}
_lock = threading.Lock()


def add_tokens(workspace_dir_name: str, tokens: int) -> None:
    """Add output tokens to a workspace's count."""
    if workspace_dir_name is None:
        raise ValueError("workspace_dir_name cannot be None")
    with _lock:
        _output_token_counts[workspace_dir_name] = (
            _output_token_counts.get(workspace_dir_name, 0) + tokens
        )


def add_input_tokens(workspace_dir_name: str, tokens: int) -> None:
    """Add input tokens to a workspace's count."""
    if workspace_dir_name is None:
        raise ValueError("workspace_dir_name cannot be None")
    with _lock:
        _input_token_counts[workspace_dir_name] = (
            _input_token_counts.get(workspace_dir_name, 0) + tokens
        )


def get_tokens(workspace_dir_name: str) -> int:
    """Get accumulated output tokens for a workspace."""
    if workspace_dir_name is None:
        raise ValueError("workspace_dir_name cannot be None")
    with _lock:
        return _output_token_counts.get(workspace_dir_name, 0)


def get_input_tokens(workspace_dir_name: str) -> int:
    """Get accumulated input tokens for a workspace."""
    if workspace_dir_name is None:
        raise ValueError("workspace_dir_name cannot be None")
    with _lock:
        return _input_token_counts.get(workspace_dir_name, 0)


def cleanup_session(workspace_dir_name: str) -> None:
    """Remove workspace from tracking."""
    with _lock:
        _output_token_counts.pop(workspace_dir_name, None)
        _input_token_counts.pop(workspace_dir_name, None)
