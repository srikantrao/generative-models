from generative_models.experiments.results import ExperimentResult
from generative_models.seeds import seed_everything


def main() -> None:
    seed_everything(7)

    result = ExperimentResult(
        name="smoke_test",
        claim="The experiment result contract can save structured evidence.",
        seed=7,
        config={
            "model": "smoke-model",
            "dataset": "smoke-dataset",
            "notes": "This is a contract smoke test, not a model run.",
        },
        metrics={
            "example_metric": 1.0,
        },
        artifacts=[],
        failure_notes=[
            "No model was trained.",
            "This only verifies the result-writing path.",
        ],
    )

    result.save_json("runs/smoke_test/result.json")


if __name__ == "__main__":
    main()
