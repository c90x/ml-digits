# Web Demo

SvelteKit browser demo for `ml-digits`. The app lets users draw a digit on a canvas, converts that drawing to a `1 x 28 x 28` tensor, and runs selected ONNX models with ONNX Runtime Web.

## Related Docs

- [Repository overview](../README.md)
- [Experiment details](../docs/experiments.md)

## Requirements

- [Bun](https://bun.sh/) for the locked install workflow in this project.

## Installing

```bash
cd web
bun install
```

## Running Locally

```bash
cd web
bun run dev
```

Vite prints the local development URL after startup.

## Quality Checks

```bash
cd web
bun run check
bun run lint
bun run format
```

## Building

```bash
cd web
bun run build
```

Preview the production build:

```bash
cd web
bun run preview
```
