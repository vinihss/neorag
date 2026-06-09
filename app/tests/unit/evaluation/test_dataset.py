import tempfile

from neorag.evaluation.dataset import EvalDataset, EvalSample, SAMPLE_DATASET


class TestEvalSample:
    def test_create_sample(self):
        sample = EvalSample(question="Q?", ground_truth="A.")
        assert sample.question == "Q?"
        assert sample.ground_truth == "A."
        assert sample.contexts == []
        assert sample.answer == ""


class TestEvalDataset:
    def test_to_dicts(self):
        dataset = EvalDataset(
            samples=[
                EvalSample(question="Q1", ground_truth="A1"),
                EvalSample(question="Q2", ground_truth="A2"),
            ]
        )
        dicts = dataset.to_dicts()
        assert len(dicts) == 2
        assert dicts[0]["question"] == "Q1"

    def test_from_dicts(self):
        data = [
            {"question": "Q?", "ground_truth": "A.", "contexts": ["ctx"], "answer": "resp"},
        ]
        dataset = EvalDataset.from_dicts(data)
        assert len(dataset.samples) == 1
        assert dataset.samples[0].contexts == ["ctx"]

    def test_save_and_load(self):
        dataset = EvalDataset(
            samples=[EvalSample(question="Q?", ground_truth="A.")]
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            dataset.save(f.name)
            loaded = EvalDataset.load(f.name)
        assert len(loaded.samples) == 1
        assert loaded.samples[0].question == "Q?"

    def test_sample_dataset_has_samples(self):
        assert len(SAMPLE_DATASET.samples) == 2
        assert "RAG" in SAMPLE_DATASET.samples[0].question
