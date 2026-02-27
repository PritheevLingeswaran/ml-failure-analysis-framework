# Contributing

## Principles
- Prefer correctness and auditability over cleverness.
- Every metric must define:
  - What it measures
  - When it lies
  - How it can be gamed
- Slicing is a first-class feature. Do not “just add overall metrics”.

## Dev workflow
```bash
make setup
make test
make lint
make run_eval
```

## Pull requests
- Include tests for new functionality.
- Update docs when changing evaluation semantics.
