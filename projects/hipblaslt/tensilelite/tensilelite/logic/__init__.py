def main():
    """Run TensileLogic without importing its heavy dependencies eagerly."""
    from .run import main as run

    return run()
