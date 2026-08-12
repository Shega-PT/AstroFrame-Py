"""Fixtures partilhadas."""

from __future__ import annotations

import pytest

from tests.helpers import make_disk_image, make_noisy_image


@pytest.fixture
def disk_image():
    return make_disk_image()


@pytest.fixture
def noisy_image():
    return make_noisy_image()
