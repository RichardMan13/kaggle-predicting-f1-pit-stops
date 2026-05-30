# Agent Configurations

## Agent skills

### Issue tracker

Issues and PRDs for this repo live as local markdown files under `.scratch/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Triage roles are mapped to standard local markdown frontmatter tags. See `docs/agents/triage-labels.md`.

### Domain docs

The codebase operates in a single global context. See `docs/agents/domain.md`.

## Rules

- **Never use emojis!** Under no circumstances should the agent use emojis in its responses, documentation, file edits, issue titles, comments, or pull requests. Keep all communications strictly professional and text-based.

## Execution Environment

The project runs under the Conda environment named `cookiecutter-kaggle`. All automation tasks must be executed via `invoke` commands (e.g., `inv clean`, `inv check`, `inv train`).

## Verification Protocol

Before completing any task, the agent must run:
- `inv check` to ensure code formatting and static analysis (linting) are clean.
- Unit tests (if applicable) to prevent regressions.

## Git Conventions

- Commit messages must be clear, professional, and written in English.
- Do not perform force pushes (`git push --force`) to remote branches.
- Create feature branches from `main` when implementing new tasks.

## User Context

- **Human Developer:** Richard (GitHub: RichardMan13)
- **Primary Language:** Portuguese (for dialogue) and English (for code, comments, and documentation).
