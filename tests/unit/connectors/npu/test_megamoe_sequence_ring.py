# Copyright (c) 2026 Huawei Technologies Co., Ltd. Licensed under CANN OSL v2.
"""Exercise the device's actual arithmetic on the host, independently of CANN."""

import importlib.util
import sys
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

SUPPORT = Path(__file__).resolve().parents[4] / "csrc/npu/ascend_kernels/megamoe_utils"


class SequenceRingTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("c++"), "requires a host C++ compiler")
    def test_device_predicate_over_four_laps(self):
        source = r"""
#define __aicore__
#include "megamoe_sequence_ring.h"
#include <cassert>
using namespace CamMegamoe;
int main()
{
    const int limit = SPLIT_SEQUENCE_MAX;
    auto wire = [limit](int n) { return (n - 1) % limit + 1; };
    for (int n = 1; n <= 4 * limit; ++n) {
        for (bool previous : {false, true}) {
            const int target = n - int(previous);
            assert(SplitSequenceReached(0, wire(n), previous) ==
                   (previous && wire(n) == 1));
            for (int observed = 1; observed <= 5 * limit; ++observed) {
                const int skew = observed - target;
                if (skew < -limit / 2 || skew > limit / 2) { continue; }
                assert(SplitSequenceReached(wire(observed), wire(n), previous) ==
                       (skew >= 0 && skew < limit / 2));
            }
        }
    }
}
"""
        with tempfile.TemporaryDirectory() as directory:
            exe = Path(directory) / "sequence_ring"
            subprocess.run(
                [
                    "c++",
                    "-std=c++11",
                    "-include",
                    "initializer_list",
                    "-I",
                    str(SUPPORT),
                    "-x",
                    "c++",
                    "-",
                    "-o",
                    str(exe),
                ],
                input=source,
                text=True,
                check=True,
            )
            subprocess.run([str(exe)], check=True)


class ControlLayoutTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("c++"), "requires a host C++ compiler")
    def test_python_allocation_contains_device_control_and_payload(self):
        """The device offsets must agree with the Python registered-window ABI."""
        path = SUPPORT.parents[3] / "afd_plugin/compat/npu/megamoe/common/megamoe_split_layout.py"
        spec = importlib.util.spec_from_file_location("taken_layout_test", path)
        layout = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = layout
        try:
            spec.loader.exec_module(layout)
        finally:
            del sys.modules[spec.name]
        source = r"""
#include "megamoe_control_layout.h"
#include <cassert>
#include <iostream>
int main() {
    using namespace CamMegamoe;
    for (int ranks = 2; ranks <= 768; ++ranks) {
        auto end = SplitTakenOffset(ranks - 1, SPLIT_TAKEN_SLOTS - 1) + 64;
        assert(end <= SplitProgressOffset(ranks));
        assert(SplitProgressOffset(ranks) + SPLIT_PROGRESS_BYTES <= SplitControlBytes(ranks));
        std::cout << ranks << " " << SPLIT_PEER_BYTES << " " << SplitControlBytes(ranks) << "\n";
    }
}
"""
        with tempfile.TemporaryDirectory() as directory:
            exe = Path(directory) / "control_layout"
            subprocess.run(["c++", "-std=c++11", "-I", str(SUPPORT), "-x", "c++",
                            "-", "-o", str(exe)], input=source, text=True, check=True)
            rows = subprocess.check_output([str(exe)], text=True).splitlines()
        for row in rows:
            ranks, peer_bytes, control_bytes = map(int, row.split())
            geometry = layout.SplitLayout(ranks - 1, 1, 1, 16, 4096, 1)
            self.assertEqual(geometry.peer_bytes, peer_bytes)
            self.assertEqual(dict(geometry.sections)["control"], control_bytes)
            self.assertEqual(geometry.buffer_bytes, sum(size for _, size in geometry.sections))
        self.assertEqual(layout.SplitLayout(4, 4, 256, 16, 4096, 6).control_bytes, 108 * 1024 + 512)
