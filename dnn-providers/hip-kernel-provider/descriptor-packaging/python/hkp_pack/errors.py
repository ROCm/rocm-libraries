class HkpPackError(Exception):
    """Raised for any recoverable, user-facing packaging failure.

    The CLI maps this to a non-zero exit and prints str(exc); tests match on
    the message text, so messages must be stable and descriptive.
    """
