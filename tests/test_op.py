"""
Test Order Parameter handling functionality.

Currently focused on testing opconvertor realted code because it is important for
OP data visualization.

Test data is stored in `./ToyData/Simulations.2`

-------------------------------------------------------------------------------
NOTE: globally import of fairmd-lipids is **STRICTLY FORBIDDEN** because it
      breaks the substitution of global path folders
"""

import copy
import os
import sys
import warnings
import numpy as np
import pytest
import pytest_check as check

# run only on sim2 mocking data
pytestmark = [pytest.mark.sim2, pytest.mark.min]


class TestBuildNiceOPdict:
    @pytest.fixture
    def systems(self):
        from fairmd.lipids.core import initialize_databank

        return initialize_databank()

    def test_work_empty(self):
        from fairmd.lipids.auxiliary.opconvertor import build_nice_OPdict
        from fairmd.lipids.molecules import Lipid

        lipid = Lipid("POPE")
        lipid.register_mapping()
        rdict = build_nice_OPdict({}, lipid)
        assert rdict == {}

    def test_build_fragmented(self, systems):
        from fairmd.lipids.auxiliary.opconvertor import build_nice_OPdict
        from fairmd.lipids.api import get_OP

        sys = systems.loc(281)
        opdata = get_OP(sys)
        rdict = build_nice_OPdict(opdata["POPC"], sys.lipids["POPC"])
        assert isinstance(rdict, dict)  # dict expected
        assert "sn-1" in rdict
        assert "sn-2" in rdict  # fragments at the top level

    def test_cnames_pl(self, systems):
        from fairmd.lipids.auxiliary.opconvertor import build_nice_OPdict
        from fairmd.lipids.api import get_OP

        sys = systems.loc(281)
        opdata = get_OP(sys)
        rdict = build_nice_OPdict(opdata["POPC"], sys.lipids["POPC"])

        # C check numbers
        def has_c(cname: str, flist: dict) -> bool:
            return any(_c["C"] == cname for _c in flist)

        check.is_true(has_c("2", rdict["sn-1"]))
        check.is_true(has_c("16", rdict["sn-1"]))
        check.is_false(has_c("1", rdict["sn-1"]))
        check.is_false(has_c("17", rdict["sn-1"]))
        # H check names
        check.is_true(all(_c["H"] in ["1", "2", "3"] for _c in rdict["sn-1"]))
        check.is_true(all(_c["H"] in ["1", "2", "3"] for _c in rdict["sn-2"]))
        # check backbone
        check.is_true(has_c("g1", rdict["glycerol backbone"]))
        check.is_true(has_c("g2", rdict["glycerol backbone"]))
        check.is_true(has_c("g3", rdict["glycerol backbone"]))

    # Check both common CI indicators
    IS_CI = os.getenv("GITHUB_ACTIONS") == "true" or os.getenv("CI") == "true"

    @pytest.mark.skipif(IS_CI, reason="This test is skipped on GitHub Actions")
    def test_build_nice_OPdict_with_all_experiments(self, monkeypatch, tmpdir):
        """Test build_nice_OPdict with all available experiments from ExperimentCollection.

        This test:
        1. Loads all OP experiments from the databank
        2. For each experiment and lipid in that experiment
        3. Runs build_nice_OPdict on the raw OP data
        4. Validates the output structure and content
        """
        import fairmd.lipids
        from fairmd.lipids.experiment import ExperimentCollection, ExperimentError
        from fairmd.lipids.auxiliary.opconvertor import build_nice_OPdict

        # Set the data path to the test data directory

        fairmd.lipids.FMDL_DATA_PATH = "../../BilayerData"
        experiments = ExperimentCollection.load_from_data("OPExperiment")

        # Verify we loaded experiments
        assert len(experiments) > 0, "No experiments were loaded"

        # Test build_nice_OPdict on all experiments
        results_count = 0
        for exp in experiments:
            # Get experiment data, but skip if no experiment data is present
            try:
                exp_data = exp.data
            except ExperimentError:
                continue
            assert isinstance(exp_data, dict), f"Experiment {exp.exp_id} data is not a dict"

            # Process each lipid in the experiment
            for lipid_name, raw_op_data in exp_data.items():
                # Get the lipid object from the experiment
                lipid = exp.lipids[lipid_name]

                # Run build_nice_OPdict
                nice_op_dict = build_nice_OPdict(raw_op_data, lipid)

                # Validate output structure
                assert isinstance(nice_op_dict, dict), (
                    f"build_nice_OPdict output is not a dict for {lipid_name} in {exp.exp_id}"
                )

                # If there's data, validate the structure of fragments
                if nice_op_dict:
                    for fragment_name, fragment_data in nice_op_dict.items():
                        assert isinstance(fragment_data, list), f"Fragment {fragment_name} data is not a list"

                        # Validate each entry in the fragment
                        for entry in fragment_data:
                            assert isinstance(entry, dict), f"Fragment entry is not a dict"
                            assert "C" in entry, "Entry missing 'C' key"
                            assert "H" in entry, "Entry missing 'H' key"
                            assert "OP" in entry, "Entry missing 'OP' key"
                            assert "STD" in entry, "Entry missing 'STD' key"

                            # Validate data types
                            assert isinstance(entry["C"], str), "C atom name is not string"
                            assert isinstance(entry["H"], str), "H atom name is not string"
                            assert isinstance(entry["OP"], (int, float)) or entry["OP"] is None, "OP is not numeric"
                            assert isinstance(entry["STD"], (int, float)) or entry["STD"] is None, "STD is not numeric"

                results_count += 1

        # Ensure we actually tested something
        assert results_count > 0, "No lipid data was tested"

        # Clean up
        ExperimentCollection.clear_instance()
