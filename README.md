# Emergent Tool Specialization

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/XXXX.XXXXX)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> 🔧 **LLM agents that spontaneously specialize in different tools through competitive selection.**

## Overview

This repository implements **Emergent Tool Specialization** — a framework where populations of identical LLM agents develop specialized tool expertise through competition alone, without explicit role assignment.

Unlike [Paper 1 (NichePopulation)](https://github.com/HowardLiYH/NichePopulation) which demonstrated emergent specialization with synthetic rules, and [Paper 2 (Emergent-Prompt-Evolution)](https://github.com/HowardLiYH/Emergent-Prompt-Evolution) which showed preference specialization in LLM agents, this work extends to **real, practical tools**:

| Level | Tool | Capability | Implementation |
|-------|------|------------|----------------|
| L0 | Base LLM | Text completion | Gemini 2.5 Flash |
| L1 | Code | Python execution | Gemini Code Execution API |
| L2 | Vision | Image analysis | Gemini Vision API |
| L3 | RAG | Document retrieval | LlamaIndex + ChromaDB |
| L4 | Web | Real-time search | Tavily API |

## Key Results

| Metric | Value | Significance |
|--------|-------|--------------|
| **Specialist Advantage** | +83.3% | p < 10⁻⁷ |
| **Vision Gap** | 8% → 88% | +80 points |
| **Code Gap** | 0% → 100% | +100 points |
| **Competition Necessity** | Proven | 0 specialists without |

## The Emergent Specialization Series

This is **Paper 3** in the Emergent Specialization research series:

| Paper | Focus | Domain | Repository |
|-------|-------|--------|------------|
| Paper 1 | Learner Populations | Time Series (Rule-based) | [NichePopulation](https://github.com/HowardLiYH/NichePopulation) |
| Paper 2 | Preference Specialization | Synthetic Rules (LLM) | [Emergent-Prompt-Evolution](https://github.com/HowardLiYH/Emergent-Prompt-Evolution) |
| **Paper 3** | **Tool Specialization** | **Real Tools (LLM)** | **This repo** |

## Architecture

```
Emergent-Tool-Specialization/
├── core/                   # CSE Algorithm
│   ├── thompson.py        # Thompson Sampling for tool selection
│   ├── fitness.py         # Fitness sharing (1/√n penalty)
│   ├── competition.py     # Competition loop
│   └── agent.py           # Specialist agent
├── tools/                  # Real Tool Implementations
│   ├── code.py            # L1: Gemini Code Execution
│   ├── vision.py          # L2: Gemini Vision
│   ├── rigorous_rag.py    # L3: LlamaIndex + ChromaDB
│   └── orchestrator.py    # L5: LangGraph (future)
├── mcp/                    # Model Context Protocol
│   ├── server.py          # Tool server
│   ├── client.py          # Agent client
│   └── schemas.py         # Tool definitions
├── memory/                 # 4-Layer Memory System
│   ├── working.py         # In-context memory
│   ├── episodic.py        # Episode storage
│   ├── semantic.py        # Compressed patterns
│   └── procedural.py      # Tool strategies
├── safety/                 # Safety & Monitoring
│   ├── collusion.py       # Collusion detection
│   └── calibration.py     # Confidence calibration
├── experiments/            # Experiment scripts
│   ├── training/          # Competition training
│   ├── ablations/         # Component ablations
│   └── phase1-3/          # Validation phases
└── results/               # Experimental results
```

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/HowardLiYH/Emergent-Tool-Specialization.git
cd Emergent-Tool-Specialization

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your keys:
# GEMINI_API_KEY=your_key
# TAVILY_API_KEY=your_key
```

### Run Training

```bash
python -m experiments.training.run_training_v2 --seed 42 --generations 100
```

### Run Evaluation

```bash
python -m experiments.phase1.run_phase1_tests
python -m experiments.ablations.run_ablations
```

## Key Algorithms

### Thompson Sampling for Tool Selection

Agents maintain Beta distribution beliefs over tool effectiveness:
```
θ_{a,r,t} ~ Beta(α, β)
tool = argmax_t sample(θ_{a,r,t})
```

### Fitness Sharing

Prevents niche crowding with penalty:
```
penalty(n) = 1/√n
```
where n = number of specialists in regime.

### Competition Loop

1. Sample regime from non-uniform distribution
2. Select K=3 competitors
3. Each agent selects tool via Thompson Sampling
4. Winner updates beliefs and memory
5. Apply fitness sharing penalty

## Theoretical Foundation

**Theorem 4 (Non-Uniform Equilibrium):**
```
n_r ∝ (f_r × R_r × D_r)^{2/3}
```
where f=frequency, R=reward, D=difficulty.

## Citation

```bibtex
@article{li2026tool,
  title={Emergent Tool Specialization in LLM Agent Populations Through Competitive Selection},
  author={Li, Yuhao},
  journal={arXiv preprint},
  year={2026}
}
```

## Related Work

- **Paper 1**: [NichePopulation](https://github.com/HowardLiYH/NichePopulation) - Emergent specialization in learner populations
- **Paper 2**: [Emergent-Prompt-Evolution](https://github.com/HowardLiYH/Emergent-Prompt-Evolution) - Preference specialization in LLM agents

## License

MIT License - see [LICENSE](LICENSE) file.

## Acknowledgments

This work builds upon the theoretical foundations established in Paper 1 (NichePopulation) and Paper 2 (Emergent-Prompt-Evolution).
