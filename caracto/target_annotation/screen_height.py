import warnings


def min_screen_height(default_height: int = 1080) -> int:
    """A method to obtain the screen height for optimized annotation interface scaling.
    In case of multiple screens, the smallest screen height is chosen.

    Args:
        default_height (int, optional): The height in pixels to default to if the screen
        height was not obtainable. Defaults to 1080.

    Returns:
        int: Screen height in pixels.
    """
    try:
        import mss
    except ImportError:
        message = (
            f"'mss' package not found, can't optimize interface to monitor height. "
            f"Using default value: {default_height}"
        )
        warnings.warn(message)
        return default_height
    sct = mss.mss()
    min_height = 0
    for monitor in sct.monitors:
        if (min_height <= 0) or (monitor["height"] > min_height):
            min_height = monitor["height"]

    if min_height <= 0:  # in case something weird happens
        return default_height

    return min_height


def main():
    print(f"Minimum screen height found is: {min_screen_height()} px.")


if __name__ == "__main__":
    main()
