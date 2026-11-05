"""build_core(core=None) — the workspace core.

Cross-repo convention: compose the cores of imported repos (inheriting
their registered processes/types), then register THIS repo's own types,
processes, and composites — so a downstream importer gets everything by
calling this build_core.
"""
from process_bigraph import allocate_core
from viva_superpowers.core_compose import (
    compose_import_cores,
    register_package_processes,
)


def build_core(core=None):
    core = core if core is not None else allocate_core()
    # 1) Compose imported repos' cores. Standard pbg-* imports
    #    auto-discover via allocate_core once installed; list any import
    #    whose processes a composite instantiates DIRECTLY so they still
    #    register by name, e.g.:
    #    compose_import_cores(core, ["viva_munk", "some_dep.core"])
    # 2) Register THIS repo's own process/step classes as first-class,
    #    browsable Registry entries (a composite that instantiates a
    #    process directly does NOT auto-register it by name). No-op until
    #    viva_hybrid/processes/ exists.
    register_package_processes(core, "viva_hybrid.processes")
    #    core.register_types({...})  # this repo's own types
    return core
