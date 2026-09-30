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


# The spelling of this name is load-bearing: bandit's B506 check clears a yaml.load() call
# only when the loader argument is named SafeLoader or CSafeLoader, so importing under either
# of those names lets the scan verify every call site rather than trusting a suppression
# comment there (SEC-00404).
try:
    from yaml import CSafeLoader as SafeLoader
except ImportError:
    from yaml import SafeLoader

try:
    from yaml import CSafeDumper as yamlDumper
except ImportError:
    from yaml import SafeDumper as yamlDumper


try:
    import joblib
except:
    warnings.warn("Missing dependency 'joblib', program will run without parallelism")
    joblib = None
