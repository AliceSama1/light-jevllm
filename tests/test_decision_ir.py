import unittest
from jev_runtime.decision_ir import BatchKey, CalibrationSpec, CompiledQuestion, DecisionRequest, ExecutionBatch, ReadoutPlan, ReadoutSlot
from jev_runtime.heads.readouts import CandidateHeadReadout, readout_for, stable_softmax


def plan(mode='candidate_head'):
    return ReadoutPlan(mode=mode, slots=(ReadoutSlot('q', 0),), labels=('a', 'b'), calibration=CalibrationSpec())


class IRTests(unittest.TestCase):
    def test_probability_and_strict_readout(self):
        self.assertAlmostEqual(sum(stable_softmax([1000, 999])), 1)
        result = CandidateHeadReadout().execute([[2, 0]], plan())[0]
        self.assertEqual(result.candidate_ids, ('a', 'b'))
        with self.assertRaises(ValueError):
            CandidateHeadReadout().execute([[1]], plan())
        with self.assertRaises(ValueError):
            CandidateHeadReadout().execute({}, plan())
        with self.assertRaises(ValueError):
            readout_for(plan('diffusion_canvas'))

    def test_shallow_freeze_and_batch_compatibility_not_content_identity(self):
        nested = []
        request = DecisionRequest('demo', 'state', (CompiledQuestion('q', 'choice'),), options={'nested': nested})
        with self.assertRaises(TypeError):
            request.options['new'] = 1
        nested.append(1)
        self.assertEqual(request.options['nested'], [1])
        key = BatchKey('demo', 'native', 'candidate_head')
        a = ExecutionBatch('demo', 'native', 'first input', plan(), key)
        b = ExecutionBatch('demo', 'native', 'different input', plan(), key)
        self.assertEqual(a.batch_key, b.batch_key)
        with self.assertRaises(ValueError):
            ExecutionBatch('other', 'native', object(), plan(), key)
