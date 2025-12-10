#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_locked_dict.py

Tests for the LockedDict class and its integration with ProspectorParams.
"""

import copy
import pickle
import numpy as np
import pytest
from prospect.models import ProspectorParams, LockedDict, priors


class TestLockedDict:
    def test_basic_locking(self):
        """Test that LockedDict prevents modification when locked."""
        d = LockedDict({"a": 1})
        d._locked = True

        # Read access should work
        assert d["a"] == 1

        # Write access should fail
        with pytest.raises(RuntimeError, match="Model configuration is locked"):
            d["a"] = 2

        with pytest.raises(RuntimeError, match="Model configuration is locked"):
            d["b"] = 3

        with pytest.raises(RuntimeError, match="Model configuration is locked"):
            del d["a"]

        with pytest.raises(RuntimeError, match="Model configuration is locked"):
            d.update({"a": 4})

    def test_unlocking(self):
        """Test that LockedDict allows modification when unlocked."""
        d = LockedDict({"a": 1})
        d._locked = False

        d["a"] = 2
        assert d["a"] == 2

        d["b"] = 3
        assert d["b"] == 3

        del d["a"]
        assert "a" not in d

        d.update({"c": 4})
        assert d["c"] == 4

    def test_deepcopy(self):
        """Test that deepcopy returns a standard dict."""
        d = LockedDict({"a": {"b": 1}})
        d._locked = True

        d_copy = copy.deepcopy(d)

        assert type(d_copy) is dict
        assert d_copy["a"]["b"] == 1

        # Modification of copy should be allowed
        d_copy["a"]["b"] = 2
        assert d_copy["a"]["b"] == 2

        # Original should be untouched
        assert d["a"]["b"] == 1

    def test_pickle(self):
        """Test that pickling returns a standard dict."""
        d = LockedDict({"a": 1})
        d._locked = True

        serialized = pickle.dumps(d)
        deserialized = pickle.loads(serialized)

        assert type(deserialized) is dict
        assert deserialized["a"] == 1


class TestProspectorParamsLocking:
    def setup_method(self):
        self.config = {
            "mass": {
                "N": 1,
                "isfree": True,
                "init": 10.0,
                "prior": priors.TopHat(mini=0, maxi=20),
            },
            "zred": {
                "N": 1,
                "isfree": False,
                "init": 0.5,
            },
        }
        self.model = ProspectorParams(self.config)

    def test_config_is_locked(self):
        """Test that the model configuration is locked after initialization."""
        # Top level dict
        assert isinstance(self.model.config_dict, LockedDict)
        assert self.model.config_dict._locked is True

        # Child dict
        assert isinstance(self.model.config_dict["mass"], LockedDict)
        assert self.model.config_dict["mass"]._locked is True

        # Attempt to modify
        with pytest.raises(RuntimeError, match="Model configuration is locked"):
            self.model.config_dict["mass"]["init"] = 11.0

        with pytest.raises(RuntimeError, match="Model configuration is locked"):
            self.model.config_dict["new_param"] = {}

    def test_modify_config_context(self):
        """Test the modify_config context manager."""
        with self.model.modify_config():
            # Should be unlocked
            assert self.model.config_dict._locked is False
            assert self.model.config_dict["mass"]._locked is False

            # Modification allowed
            self.model.config_dict["mass"]["init"] = 11.0

        # Should be re-locked
        assert self.model.config_dict._locked is True
        assert self.model.config_dict["mass"]._locked is True

        # Check value persisted
        assert self.model.config_dict["mass"]["init"] == 11.0

        # Check that params were updated (configure is re-run)
        # Note: init value in config only affects params if we reset or if logic uses it.
        # configure(reset=False) doesn't overwrite existing params unless we manually do it.
        # But let's check if the model state is consistent.

        # If we change something structural like N, it should update.

    def test_add_dependency_via_modify(self):
        """Test adding a dependency dynamically."""

        def new_dep(mass=0, **kwargs):
            return mass / 10.0

        with self.model.modify_config():
            self.model.config_dict["zred"]["depends_on"] = new_dep

        # Dependency should now be active
        self.model.set_parameters(np.array([10.0]))  # mass = 10
        assert self.model.params["zred"][0] == 1.0  # 10 / 10

        self.model.set_parameters(np.array([20.0]))  # mass = 20
        assert self.model.params["zred"][0] == 2.0  # 20 / 10

    def test_structural_change(self):
        """Test changing a parameter from fixed to free."""

        assert "zred" not in self.model.free_params

        with self.model.modify_config():
            self.model.config_dict["zred"]["isfree"] = True
            self.model.config_dict["zred"]["prior"] = priors.TopHat(mini=0, maxi=5)

        # Check update
        assert "zred" in self.model.free_params
        assert self.model.ndim == 2
        assert len(self.model.theta_index) == 2

        # config_list should be synced
        zred_cfg_list = next(x for x in self.model.config_list if x["name"] == "zred")
        assert zred_cfg_list["isfree"] is True

    def test_pickle_compatibility(self):
        """Test that the model can be pickled and unpickled, and what state it ends up in."""
        serialized = pickle.dumps(self.model)
        deserialized = pickle.loads(serialized)

        # As per design, it should deserialize to a standard dict for config_dict
        assert type(deserialized.config_dict) is dict
        assert type(deserialized.config_dict["mass"]) is dict

        # But functionality should work
        assert deserialized.ndim == self.model.ndim

        # modify_config should handle plain dicts gracefully (or just work on them)
        with deserialized.modify_config():
            deserialized.config_dict["mass"]["init"] = 12.0

        # After exiting modify_config, it should be RE-LOCKED because modify_config calls configure()
        # which calls _lock_configuration()
        assert isinstance(deserialized.config_dict, LockedDict)
        assert deserialized.config_dict._locked is True
