"""Compatibility server module expected by some OpenEnv validators."""

from main import app, main as _main


def main():
    """Validator-expected callable entry point."""
    _main()


if __name__ == "__main__":
    main()

