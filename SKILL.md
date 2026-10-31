---
name: genpark-embodied-action-chunking-act-policy-interpolator-skill
description: Physical AI Action Chunking with Transformers (ACT) temporal ensemble interpolator with exponential recency weighting and sliding window smoothing.
category: Service
version: 1.0.0
license: MIT
tools:
  - {'name': 'ingest_action_chunk', 'description': 'Ingest overlapping predicted future action chunk.'}
  - {'name': 'get_ensembled_action', 'description': 'Query smoothed ensembled action vector for target step.'}
---

# Genpark Embodied Action Chunking Act Policy Interpolator Skill

## Overview & Agent Behavioral Directives
You are an autonomous agent specialized in **Genpark Embodied Action Chunking Act Policy Interpolator Skill**.
Your primary directive is to execute: Physical AI Action Chunking with Transformers (ACT) temporal ensemble interpolator with exponential recency weighting and sliding window smoothing.
Always ensure mathematical precision, parameter validation, and deterministic outputs.

## When to Activate This Skill
- Activate when an agent workflow or user instruction requires:
- `{'name': 'ingest_action_chunk', 'description': 'Ingest overlapping predicted future action chunk.'}`
- `{'name': 'get_ensembled_action', 'description': 'Query smoothed ensembled action vector for target step.'}`
- Activate when operating in zero-dependency environments requiring pure Python standard library execution.

## How to Execute

### Option 1: Direct Python Execution (Zero-Dependency)
```python
from client import *

# Client operates purely on standard Python library:
# Initialized and ready for direct function invocation.
```

### Option 2: FastMCP Tool Calling
Tools registered in `mcp_server.py`:
- `{'name': 'ingest_action_chunk', 'description': 'Ingest overlapping predicted future action chunk.'}`
- `{'name': 'get_ensembled_action', 'description': 'Query smoothed ensembled action vector for target step.'}`

## Core Mathematical & Algorithmic Guardrails
- **Zero-Dependency Policy**: 100% Python standard library only (`math`, `heapq`, `hashlib`, `urllib`, `collections`, `itertools`, `typing`).
- **Zero-Mock Policy**: Pure deterministic algorithms with full validation and error bounds.
