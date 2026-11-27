class HkpPackError(Exception):
    """Raised for any recoverable, user-facing packaging failure.

    The CLI maps this to a non-zero exit and prints str(exc); tests match on
    the message text, so messages must be stable and descriptive.
    """


class OpSchemaRegistryError(HkpPackError):
    """Raised when the op-schema registry a UED pattern resolves against cannot be
    used: it is unavailable, unreadable or malformed. A pattern that names an
    opcode the registry lacks is an ordinary HkpPackError.
    """
