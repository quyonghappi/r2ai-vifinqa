"""Post-inference repair pass: scan predictions/failures, diagnose, repair, validate, write.

See POST_INFERENCE_REPAIR_DESIGN.md for the design this module implements. Nothing here mutates
``predictions/`` or ``failures/`` in place -- see ``repair.common`` for the atomic-write/preserve
conventions every entry point follows.
"""
