import warnings

def showwarning(message, category, filename, lineno, file=None, line=None):
    msg = f"> {category.__name__}: {message}"
    if TENSILE_TERM_COLORS:
        msg = f"[yellow]{msg}[/yellow]"
    print(msg)

warnings.showwarning = showwarning

TENSILE_TERM_COLORS: bool = False
try:
    from rich import print as print
    TENSILE_TERM_COLORS = True
except ImportError:
    print = print


try:
    from yaml import CSafeLoader as yamlLoader
except ImportError:
    from yaml import SafeLoader as yamlLoader

try:
    from yaml import CSafeDumper as yamlDumper
except ImportError:
    from yaml import SafeDumper as yamlDumper


try:
    import joblib
except:
    warnings.warn("Missing dependency 'joblib', program will run without parallelism")
    joblib = None
