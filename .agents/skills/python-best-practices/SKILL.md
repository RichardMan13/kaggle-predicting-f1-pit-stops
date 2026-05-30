---
name: python-best-practices
description: Write clean, readable, and production-grade Python code. Use when creating new Python scripts, refactoring existing Python functions, adding type hints, writing docstrings, or ensuring PEP 8 styling compliance.
---

# Python Clean Code & Best Practices

Use this skill to write high-quality, readable, and highly maintainable Python code following industry standards and modern features.

## Quick start

Write a clean, fully typed, docstring-documented function with robust error handling and `pathlib`:
```python
from pathlib import Path
from typing import Optional

def load_dataset(file_path: Path) -> Optional[str]:
    """Loads raw text content from a given file path safely.

    Args:
        file_path: The dynamic path to the target text file.

    Returns:
        The text content if loaded successfully, otherwise None.

    Raises:
        ValueError: If file_path is not a valid Path instance.
    """
    if not isinstance(file_path, Path):
        raise ValueError("Provided file_path must be a Path object.")
        
    try:
        return file_path.read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError) as e:
        print(f"[Warning] Failed to read {file_path}: {e}")
        return None
```

## Workflows

### 1. Robust Type Hinting & PEP 8 Style
- [ ] **Type Annotations:** Annotate all function inputs and return types. Use `typing` primitives (e.g., `Optional`, `Union`, `List`, `Dict`) for complex types.
- [ ] **Formatting (Ruff / Black):** Ensure line limits of 88 (or 79) characters. Avoid excessive vertical blank lines or clustered imports.
- [ ] **Import Ordering:** Group imports clearly: standard libraries, third-party libraries, and local modules (separated by single blank lines).

### 2. Path & Resource Handling
- [ ] **Avoid os.path:** Always use `pathlib.Path` for path manipulations to guarantee OS-agnostic compatibility.
- [ ] **Context Managers:** Always use the `with` statement for resource handling (opening files, databases, locks) to ensure resources are cleaned up cleanly.
- [ ] **Explicit Encoding:** Always specify `encoding="utf-8"` when reading or writing text files.
