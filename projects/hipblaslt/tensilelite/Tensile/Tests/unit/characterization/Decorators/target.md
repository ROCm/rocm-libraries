# Characterization target — `Tensile/Utilities/Decorators/{Shared,Timing,Profile}.py`

Part of the master-plan remaining-module sweep. **After 100.00% line** on all
three (Shared 13/0, Timing 15/0, Profile 25/0). Drives `envVariableIsSet`
(truthy values + unset), `CallableGuard` (call + `__bool__` raises), `@timing`
(disabled passthrough + enabled wrapper via env), `@profile`
(disabled passthrough + enabled cProfile over tmp cwd), `initProfileArtifacts`.
Pure (env/fs via monkeypatch + tmp_path); no resistance.
