"""
Offensive side-channel attacks.

Sibling to the detector harness in the parent package. Where the detectors
(``instruments.py``, ``noninterference.py``, ``run_all.py``, ...) answer *does a
secret-dependent execution path exist, and is the compiler to blame?*, the code
here answers the follow-on question: *given a leak that does exist, what can an
attacker recover through it?*

``base`` defines the vocabulary every attack shares (``Oracle``, ``Labeler``,
``ExtractionAttack``, ``AttackResult``); each concrete attack lives in its own
sub-package (e.g. ``early_exit_gpt``).
"""
