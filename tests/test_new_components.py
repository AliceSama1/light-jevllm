"""Tests for independent media, calibration and parity components."""
import unittest
from jev_runtime.media import MediaLimits, MediaNormalizer
from jev_runtime.calibration import CalibrationPolicy
from jev_runtime.parity import ParityRunner

class ComponentTests(unittest.TestCase):
    def test_media_identity_and_limits(self):
        probe = lambda data, mime: {"width": 10, "height": 10, "frames": 2}
        result = MediaNormalizer(MediaLimits(max_bytes=3, max_pixels=100, max_frames=2), probe).normalize(b"abc", "IMAGE/PNG")
        self.assertEqual(result.metadata.digest, MediaNormalizer(probe=probe).normalize(b"abc", "image/png").metadata.digest)
        with self.assertRaises(ValueError): MediaNormalizer().normalize(b"x", "image/gif")
        with self.assertRaises(ValueError): MediaNormalizer(MediaLimits(max_pixels=2), probe).normalize(b"x", "image/png")
        with self.assertRaises(ValueError): MediaNormalizer(MediaLimits(max_frames=1), probe).normalize(b"x", "video/mp4")

    def test_calibration_temperature_and_metadata(self):
        policy = CalibrationPolicy(temperature=2, version="test", metadata={"source": "unit"})
        probabilities = policy.apply((0.0, 2.0))
        self.assertAlmostEqual(sum(probabilities), 1.0)
        self.assertEqual(policy.describe()["version"], "test")
        with self.assertRaises(ValueError): CalibrationPolicy(temperature=0)

    def test_parity_gate_and_drift(self):
        runner = ParityRunner(max_probability_drift=0.01)
        report = runner.run([1, 2], lambda _: {"choice": "a", "probabilities": {"a": .8, "b": .2}}, lambda _: {"choice": "a", "probabilities": {"a": .795, "b": .205}})
        self.assertTrue(report.passed)
        failed = runner.compare(((.9, .1),), ((.7, .3),))
        self.assertFalse(failed.passed)

if __name__ == "__main__": unittest.main()
