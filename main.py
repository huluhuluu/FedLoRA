"""Run centralized or federated LoRA experiments from JSON."""

import argparse
import json
from utils.config import load_config, validate_components
from utils.helper import configure_runtime


def run(config, resume=None):
    configure_runtime(config["runtime"]["cpu_threads"])
    validate_components(config)
    from utils.experiment import load_checkpoint
    from utils.data import load_data
    from execution import create_executor

    saved = load_checkpoint(resume, config)
    bundle = load_data(config)
    executor = create_executor(config, bundle, initialize=saved is None)
    runner, error = None, None
    try:
        if executor.is_leader:
            if config["experiment"]["mode"] == "centralized":
                from training.centralized import CentralizedTrainer

                runner = CentralizedTrainer(config, bundle, executor, resume, saved)
            else:
                from simulation.engine import Engine

                runner = Engine(config, bundle, executor, resume, saved)
            runner.run()
        else:
            executor.serve()
    except BaseException as exc:
        error = exc
        raise
    finally:
        try:
            executor.close(cancel=error is not None)
        except BaseException as exc:
            if error is None:
                error = exc
                raise
            error.add_note(f"Executor cleanup failed: {exc}")
        finally:
            if runner is not None:
                runner.experiment.status(
                    "failed" if error is not None else "completed",
                    str(error) if error is not None else None,
                )
    return runner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=JSON")
    parser.add_argument("--resume")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config, args.set)
    if args.validate_only:
        configure_runtime(config["runtime"]["cpu_threads"])
        validate_components(config, construct=True)
        print(json.dumps(config, indent=2))
    else:
        run(config, args.resume)


if __name__ == "__main__":
    main()
