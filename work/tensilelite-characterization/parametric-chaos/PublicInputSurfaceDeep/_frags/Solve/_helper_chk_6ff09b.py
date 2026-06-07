def guard_equiv(results_csv_exists: bool, force_redo: bool) -> bool:
    """post: __return__ == ((not results_csv_exists) or force_redo)"""
    return (not results_csv_exists) or force_redo
