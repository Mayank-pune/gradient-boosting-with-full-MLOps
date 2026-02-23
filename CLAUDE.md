# Project 1: Gradient Boosting from Scratch + Full MLOps Pipeline

## The Bigger Picture — 3-Project Fast Track
This project is **Project 1 of 3** in my MLOps Bootcamp. Before starting, read the full plan in `../ML_Projects_MLOps_Fast_Track_Guide.pdf` (or the .docx version in the same parent folder). It contains detailed implementation guides for all 3 projects.

### The 3-Project Roadmap (11-13 weeks total)
1. **Project 1 (this one): Gradient Boosting from Scratch** (Weeks 1-3) — Classical ML + MLOps bootcamp. Learn Git, DVC, MLflow, FastAPI, Docker, Evidently AI.
2. **Project 2: Autograd Engine + Neural Network** (Weeks 4-7) — Deep learning fundamentals + optimization deep dive (L-BFGS, constrained optimization) for quant roles. Adds GitHub Actions CI/CD and GPU-aware deployment.
3. **Project 3: Time Series Transformer from Scratch** (Weeks 8-13) — Transformers + real-time forecasting. Full production stack: Airflow orchestration, Prometheus + Grafana monitoring, automated retraining with champion-challenger pattern.

### Why This Order
- Project 1 is the MLOps bootcamp — learn the core tools on familiar ML ground
- Project 2 builds DL depth + optimization theory that quant interviews test
- Project 3 is the capstone — most complex pipeline on real-time data
- Each project reuses and extends the MLOps tools from the previous one
- Together they tell: "I understand ML from the ground up, I can build DL systems, and I can deploy production pipelines on real-time data"
- This covers DS, MLE, MLOps, and quant roles

## About Me
- **Name**: Mayank
- **Background**: Fresher, just graduated from IIIT-H
- **ML knowledge**: Basic understanding of classical ML and deep learning
- **Target roles**: Data Science, MLE, MLOps, ML Quant
- **Goal**: Build a strong portfolio of projects with production-grade MLOps pipelines
- **GitHub**: Push everything to GitHub — this is a portfolio project
- **Writing**: Plan to document this project on Medium or similar platform

## Project Goal
Build a gradient boosting classifier **entirely from scratch** (no scikit-learn GBM) using decision trees as weak learners, applied to **credit risk scoring**. Then wrap it in a complete MLOps pipeline: versioned data (DVC), tracked experiments (MLflow), REST API (FastAPI), containerized deployment (Docker), and basic monitoring (Evidently AI).

## How I Want to Learn

### Teaching Style
- **Explain concepts BEFORE writing code.** I want to understand the math, the intuition, and the "why" before implementing anything.
- **Do NOT write code for me.** Guide me, explain what needs to happen, and let me write it myself. Review my code and suggest improvements.
- **Question every choice.** For every tool, algorithm, or approach — tell me the alternatives, pros/cons, and why we're picking one over the other.
- **Line by line understanding.** If I write something, I should be able to explain every single line. If I can't, we need to slow down.

### Terminal & Git
- I want to become **very terminal-savvy**. Teach me every git command, explain flags, show me best practices.
- I will operate all git commands myself — commits, branches, PRs, everything.
- Explain git workflows (feature branches, commit message conventions, .gitignore best practices).

### Pace
- **Depth over speed.** I'd rather spend an extra day understanding something deeply than rush through it.
- Break complex topics into small, digestible pieces.
- Use analogies and real-world examples to explain abstract concepts.

## Project Structure (Planned)
```
Project1/
├── src/                    # Core algorithm code
│   ├── tree.py             # Decision tree implementation
│   ├── boosting.py         # Gradient boosting implementation
│   └── losses.py           # Loss function implementations
├── pipeline/               # Data processing + training
├── api/                    # FastAPI serving code
├── monitoring/             # Drift detection scripts
├── tests/                  # Unit + integration tests
├── notebooks/              # EDA and experimentation
├── docker-compose.yml
├── dvc.yaml                # DVC pipeline definition
├── MLproject               # MLflow project file
├── Dockerfile
├── requirements.txt
├── .gitignore
└── README.md
```

## MLOps Stack
- **Version Control**: Git + GitHub
- **Data Versioning**: DVC
- **Experiment Tracking**: MLflow
- **API Serving**: FastAPI
- **Containerization**: Docker
- **Monitoring**: Evidently AI (basic drift reports)
- **Testing**: pytest

## Phases

### Phase 1: Core Algorithm (Week 1)
- Decision tree from scratch (greedy splits, MSE reduction)
- Gradient boosting loop (residual learning)
- Loss functions (squared loss, log-loss)
- Regularization (shrinkage, max depth, subsampling, early stopping)

### Phase 2: Credit Risk Application (Week 2)
- Dataset: German Credit or Lending Club
- Feature engineering, EDA
- Evaluation: AUC-ROC, precision-recall, calibration plots
- Benchmark against scikit-learn GBM and XGBoost

### Phase 3: MLOps Pipeline (Week 3)
- DVC for data versioning
- MLflow for experiment tracking
- FastAPI for model serving
- Docker for containerization
- Evidently AI for drift monitoring

## Important Reminders
- Always explain the "why" before the "how"
- Never write code for me — guide me to write it myself
- For every decision, present alternatives with pros and cons
- Help me write clean, well-documented, production-quality code
- Remind me to commit frequently with meaningful messages
- Help me think about what would make a good Medium article
- Always help me write very modulated code, with oops concepts in Python which make me industry ready  
