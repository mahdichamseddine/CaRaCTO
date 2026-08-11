"""Screen-height detection used to scale the manual annotation interface."""

import warnings


def min_screen_height(default_height: int = 1080) -> int:
    """Return the smallest connected screen's height in pixels.

    Args:
        default_height: height in pixels to default to if the screen height
            was not obtainable.

    Returns:
        Screen height in pixels.

    """
    try:
        import mss  # noqa: PLC0415 (optional dependency, not in pyproject.toml)  # ty: ignore[unresolved-import]
    except ImportError:
        message = (
            f"'mss' package not found, can't optimize interface to monitor height. "
            f"Using default value: {default_height}"
        )
        warnings.warn(message, stacklevel=2)
        return default_height
    sct = mss.mss()
    min_height = 0
    for monitor in sct.monitors:
        if (min_height <= 0) or (monitor["height"] > min_height):
            min_height = monitor["height"]

    if min_height <= 0:  # in case something weird happens
        return default_height

    return min_height


def main() -> None:
    """Print the detected minimum screen height."""
    print(f"Minimum screen height found is: {min_screen_height()} px.")


if __name__ == "__main__":
    main()
