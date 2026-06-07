# AI QA Agent — Complete Project Reference Guide

> **Version:** 1.0.0  
> **Author:** Swapnil Chhabilwad  
> **Repository:** `https://github.com/swapnilchhabilwad/ai_qa_agent`  
> **Purpose:** Transform Product Requirement Documents (PRDs) into exhaustive, professional-grade test suites using AI.

---

## Table of Contents

1. [What Is This Project?](#1-what-is-this-project)
2. [Project Directory Tree](#2-project-directory-tree)
3. [System Architecture](#3-system-architecture)
4. [Pipeline Flow Diagram](#4-pipeline-flow-diagram)
5. [Setup & Installation](#5-setup--installation)
6. [Configuration (.env)](#6-configuration-env)
7. [How to Run the Project](#7-how-to-run-the-project)
8. [All Libraries & Dependencies](#8-all-libraries--dependencies)
9. [Complete Function Reference](#9-complete-function-reference)
10. [Data Structures / Models](#10-data-structures--models)
11. [Data Flow — Step by Step](#11-data-flow--step-by-step)
12. [Test Suite Documentation](#12-test-suite-documentation)
13. [Output Formats](#13-output-formats)
14. [Troubleshooting & Common Issues](#14-troubleshooting--common-issues)
15. [Extending the Framework](#15-extending-the-framework)
16. [Glossary](#16-glossary)

---

## 1. What Is This Project?

The **AI QA Agent** is an intelligent, context-aware test generation framework that:

- 📄 **Reads** Product Requirement Documents (PRDs) in `.txt` or `.pdf` format
- 🧠 **Analyzes** them using Azure OpenAI GPT models to extract features, rules, and edge cases
- 🔗 **Remembers** previously analyzed features across multiple PRDs (via persistent memory)
- ⚡ **Detects** functional regressions when new features impact old ones (Impact Analysis)
- 🧪 **Generates** exhaustive test cases across **21 testing categories** (Smoke, API, Security, Performance, etc.)
- 🔄 **Self-evaluates** its own coverage and iteratively fills gaps until full coverage is reached
- 📊 **Exports** to structured CSV files with billing/usage reports

### Who Is This For?

- **QA Engineers** who want to automate test case creation from requirements
- **Test Managers** who need exhaustive coverage reports
- **Developers** who want regression detection when new PRDs touch existing features
- **Anyone** tired of manually writing hundreds of test cases

---

## 2. Project Directory Tree

```
ai_qa_agent/
│
├── .gitignore                          # Git ignore rules
├── .env (NOT tracked by git)           # API keys, endpoints, config
│
├── requirements.txt                    # Python package dependencies
├── FRAMEWORK_DOCS.md                   # High-level framework overview
├── AI_QA_AGENT_COMPLETE_REFERENCE.md   # THIS FILE — complete reference
│
├── main.py                             # CLI entry point
├── app.py                              # FastAPI web server (API mode)
├── scratch_test.py                     # Standalone LLM connectivity test
├── sample.txt                          # Sample PRD file (empty placeholder)
│
├── agents/                             # 🧠 The "Brains" of the system
│   ├── __init__.py
│   ├── analyzer.py                     # Requirement + Impact Analysis
│   └── test_generator.py              # Multi-pass test generation engine
│
├── utils/                              # 🔧 Utilities & Helpers
│   ├── __init__.py
│   ├── config.py                       # Environment variable management
│   ├── loader.py                       # File loaders (txt, pdf, Confluence)
│   ├── chunking.py                     # Document chunking for LLM context
│   ├── vector_store.py                 # FAISS / Local vector search
│   ├── memory.py                       # Persistent cross-PRD memory
│   ├── exporter.py                     # CSV export
│   ├── test_suite.py                   # Data models, parsers, renderers
│   ├── local_fallback.py               # Offline keyword-based fallback
│   └── usage_tracker.py                # Token/cost billing tracker
│
├── tests/                              # ✅ Unit Tests
│   ├── __init__.py
│   ├── test_exporter.py                # Tests for CSV export
│   ├── test_generator_refinement.py    # Tests for self-evaluation loop
│   └── test_test_suite.py              # Tests for parsing/dedup/coverage
│
├── data/                               # 📦 Runtime Data (gitignored)
│   ├── system_memory.json              # Default project memory file
│   ├── acto_memory.json                # Project: acto memory
│   ├── orangehrm_memory.json           # Project: orangehrm memory
│   ├── usage_log.json                  # Daily token usage logs
│   ├── billing_history.csv             # Transactional CSV billing audit
│   │
│   ├── Projects/                       # 📂 Source PRD files
│   │   ├── Project name/
│   │   │   ├── Onboarding/
│   │   │   │   ├── prd1.txt
│   │   │   │   └── Onboarding Paid User Enterprise.txt
│   │   │   ├── Access_Management/
│   │   │   │   └── 'Disclosure Owner' Role.txt
│   │   │   └── Billing_and_Subscription/
│   │   │       └── Upgrade plan paid user enterprise.txt
│   │   ├── acto/
│   │   │   └── rtmsheet1.txt
│   │   └── orangehrm/
│   │       └── Onboarding/
│   │           └── orangehrm onboarding.txt
│   │
│   └── results/                        # 📊 Generated test case CSVs
│       ├── Project name/
│       │   ├── Onboarding/
│       │   │   ├── prd1_test_cases.csv
│       │   │   └── Onboarding Paid User Enterprise_test_cases.csv
│       │   ├── Access_Management/
│       │   │   └── 'Disclosure Owner' Role_test_cases.csv
│       │   └── Billing_and_Subscription/
│       │       └── Upgrade plan paid user enterprise_test_cases.csv
│       ├── acto/
│       │   └── rtmsheet1_test_cases.csv
│       └── orangehrm/
│           └── Onboarding/
│               └── orangehrm onboarding_test_cases.csv
│
└── embeddings/                         # 🧬 FAISS vector indices (auto-generated)
    ├── index.faiss                     # Binary FAISS index
    ├── index.pkl                       # FAISS metadata pickle
    ├── manifest.json                   # Source hash → cache validity
    └── local_store.json                # Offline keyword store
```

---

## 3. System Architecture

### High-Level Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           AI QA AGENT                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌───────────────────┐   │
│  │  CLI     │   │  FastAPI │   │ FastAPI  │   │  Output           │   │
│  │  main.py │   │  app.py  │   │ /usage   │   │  (CSV + Console)  │   │
│  └────┬─────┘   └────┬─────┘   └────┬─────┘   └───────────────────┘   │
│       │              │              │                                  │
│       └──────┬───────┘              │                                  │
│              │                      │                                  │
│              ▼                      │                                  │
│     ┌────────────────┐             │                                  │
│     │   run_agent()  │             │                                  │
│     │   (Orchestrator)│             │                                  │
│     └───────┬────────┘             │                                  │
│             │                      │                                  │
│    ┌────────┼────────────────────────────┐                            │
│    │        ▼                        │   │                            │
│    │  ┌──────────┐                   │   │                            │
│    │  │ Loader   │ ← .txt / .pdf    │   │                            │
│    │  └────┬─────┘                   │   │                            │
│    │       ▼                        │   │                            │
│    │  ┌──────────┐                   │   │                            │
│    │  │ Chunking │ → 1000-char      │   │                            │
│    │  └────┬─────┘   chunks         │   │                            │
│    │       ▼                        │   │                            │
│    │  ┌────────────────┐            │   │                            │
│    │  │ Vector Store   │ ← FAISS    │   │                            │
│    │  │ (OpenAI/local) │    or      │   │                            │
│    │  └───────┬────────┘    local   │   │                            │
│    │          ▼                     │   │                            │
│    │  ┌────────────────┐            │   │                            │
│    │  │ Similarity     │ → top 20  │   │                            │
│    │  │ Search (k=20)  │   chunks  │   │                            │
│    │  └───────┬────────┘            │   │                            │
│    │          ▼                     │   │                            │
│    │  ┌────────────────┐            │   │                            │
│    │  │ Requirement    │ ← LLM or  │   │                            │
│    │  │ Analysis       │   local   │   │                            │
│    │  └───────┬────────┘   fallback│   │                            │
│    │          ▼                     │   │                            │
│    │  ┌────────────────┐            │   │                            │
│    │  │ MemoryStore    │ ← Load    │   │                            │
│    │  │ Impact Analysis│   history │   │                            │
│    │  └───────┬────────┘            │   │                            │
│    │          ▼                     │   │                            │
│    │  ┌────────────────────────────────────────────┐                 │
│    │  │        Test Generator (test_generator.py)   │                 │
│    │  │                                            │                 │
│    │  │  Pass 1: Generate across 21 categories     │                 │
│    │  │     ├─ Initial generation                   │                 │
│    │  │     └─ Expansion generation                 │                 │
│    │  │                                            │                 │
│    │  │  Pass 2: Self-Evaluation Loop (max 10 it.) │                 │
│    │  │     ├─ Coverage Evaluation                  │                 │
│    │  │     ├─ Gap Identification                   │                 │
│    │  │     └─ Gap-Filling Generation               │                 │
│    │  │                                            │                 │
│    │  └───────────────────────┬────────────────────┘                 │
│    │                          ▼                                      │
│    │  ┌────────────────┐      │                                      │
│    │  │ Exporter       │ ← Final test suite                         │
│    │  │ (CSV)          │                                              │
│    │  └───────┬────────┘                                              │
│    │          ▼                                                      │
│    │  ┌────────────────┐                                              │
│    │  │ Usage Tracker  │ ← Token/cost logging                        │
│    │  └────────────────┘                                              │
│    └──────────────────────────────────────────────────────────────────┘
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 21 Test Output Categories

The generator produces tests organized into these buckets:

```
┌─────────────────────────────────────────────────────┐
│             21 OUTPUT TEST CATEGORIES                │
├─────────────────────────────────────────────────────┤
│  1. Foundational Smoke and Sanity                   │
│  2. End-to-End Business Workflows                   │
│  3. Core Functional Requirements                    │
│  4. Detailed Functional Rules and Constraints       │
│  5. Non-Functional Expectations                     │
│  6. Front-end UI and UX                             │
│  7. Front-end Responsive and Cross-Browser          │
│  8. Back-end Database Persistence and Integrity     │
│  9. Back-end Schema and Audit Validation            │
│ 10. API Contract and Schema Validation              │
│ 11. API CRUD and State Change Correctness           │
│ 12. Positive Happy Path Scenarios                   │
│ 13. Negative Error Handling and Validation Scenarios│
│ 14. State Transitions and Lifecycle flows           │
│ 15. User Behavior                                   │
│ 16. Concurrency and Race Conditions                 │
│ 17. Interruption and Recovery scenarios             │
│ 18. Boundary and Edge Cases                         │
│ 19. System Performance and Latency SLA              │
│ 20. Security, Auth, and Role-Based Access Control   │
└─────────────────────────────────────────────────────┘
```

---

## 4. Pipeline Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    COMPLETE PIPELINE EXECUTION FLOW                          │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  START                                                                       │
│    │                                                                         │
│    ▼                                                                         │
│  ┌─────────────────────────────────────┐                                    │
│  │ load_environment()                  │  ← Reads .env file                  │
│  │ has_valid_openai_key()              │  ← Checks AZURE_OPENAI_API_KEY      │
│  └─────────────┬───────────────────────┘                                    │
│                │                                                             │
│                ▼                                                             │
│  ┌─────────────────────────────────────┐                                    │
│  │ run_agent(query, file_path,         │  ← Main orchestrator                │
│  │           project_name)             │                                    │
│  └─────────────┬───────────────────────┘                                    │
│                │                                                             │
│     ┌──────────┴──────────┐                                                  │
│     ▼                     ▼                                                  │
│  ┌────────────┐   ┌────────────────┐                                        │
│  │load_data() │   │usage_tracker    │                                        │
│  │  ↓          │   │.set_current_   │                                        │
│  │get_configured│   │source(file_path)│                                     │
│  │_sources()   │   └────────────────┘                                        │
│  │  ↓          │                                                             │
│  │load_txt/pdf │                                                             │
│  │  ↓          │                                                             │
│  │SHA-256 hash │                                                             │
│  └──────┬──────┘                                                             │
│         ▼                                                                    │
│  ┌─────────────────────────────────────┐                                    │
│  │ init_vector_store(text, manifest)   │                                    │
│  │  ↓                                   │                                    │
│  │ chunk_text(text) → 1000-char chunks │                                    │
│  │  ↓                                   │                                    │
│  │ vector_store_matches(manifest)?      │──Yes──→ load_vector_store()        │
│  │  ↓ No                                │                                    │
│  │ create_vector_store(chunks, manifest)│                                    │
│  │  ├─ OpenAI: FAISS.from_texts()       │                                    │
│  │  └─ Local: LocalVectorStore()        │                                    │
│  └─────────────┬───────────────────────┘                                    │
│                │                                                             │
│                ▼                                                             │
│  ┌─────────────────────────────────────┐                                    │
│  │ db.similarity_search(query, k=20)   │  ← Retrieve top 20 relevant chunks │
│  └─────────────┬───────────────────────┘                                    │
│                │                                                             │
│                ▼                                                             │
│  ┌─────────────────────────────────────┐                                    │
│  │ Step 1: REQUIREMENT ANALYSIS        │                                    │
│  │ analyze_requirement(context)        │                                    │
│  │  ↓                                   │                                    │
│  │ LLM Prompt → Features, User Flows,  │                                    │
│  │              Business Rules, Edges   │                                    │
│  │ OR local_fallback.build_analysis()  │                                    │
│  └─────────────┬───────────────────────┘                                    │
│                │                                                             │
│                ▼                                                             │
│  ┌─────────────────────────────────────┐                                    │
│  │ Step 2: IMPACT ANALYSIS             │                                    │
│  │ MemoryStore(project_name)           │                                    │
│  │  ↓                                   │                                    │
│  │ memory.get_historical_context(       │                                    │
│  │   exclude_file_id=file_path)        │                                    │
│  │  ↓                                   │                                    │
│  │ analyze_impact(analysis,             │                                    │
│  │   historical_context)               │                                    │
│  │  ↓                                   │                                    │
│  │ Returns "NO_IMPACT" or impact report│                                    │
│  └─────────────┬───────────────────────┘                                    │
│                │                                                             │
│                ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ Step 3: TEST GENERATION (test_generator.py)                             ││
│  │                                                                          ││
│  │  generate_test_cases(analysis, impact_analysis)                          ││
│  │    │                                                                     ││
│  │    ├── _generate_initial_cases() ← PASS 1                               ││
│  │    │     │                                                               ││
│  │    │     ├── For each in 21 categories:                                 ││
│  │    │     │   ├── _build_generation_prompt()                             ││
│  │    │     │   ├── _invoke_llm() → LLM call                               ││
│  │    │     │   ├── _parse_generation_response() → TestCase[]              ││
│  │    │     │   ├── _build_generation_expansion_prompt()                   ││
│  │    │     │   ├── _invoke_llm() → LLM call (expansion)                   ││
│  │    │     │   └── _parse_generation_response() → TestCase[]              ││
│  │    │     └── deduplicate_test_cases()                                   ││
│  │    │                                                                     ││
│  │    ├── Pass 2 SELF-EVALUATION LOOP (max 10 iterations):                 ││
│  │    │     │                                                               ││
│  │    │     ├── _evaluate_coverage()                                       ││
│  │    │     │   ├── _build_heuristic_coverage_report() (rule-based)        ││
│  │    │     │   ├── _build_coverage_evaluation_prompt() (LLM-based)        ││
│  │    │     │   └── _merge_coverage_reports()                               ││
│  │    │     │                                                               ││
│  │    │     ├── If FULL_COVERAGE → BREAK                                   ││
│  │    │     │                                                               ││
│  │    │     └── _generate_gap_cases()                                      ││
│  │    │         ├── _build_gap_generation_prompt()                         ││
│  │    │         ├── _invoke_llm() → LLM call                               ││
│  │    │         └── _parse_generation_response()                           ││
│  │    │                                                                     ││
│  │    └── render_final_report() → Final markdown string                    ││
│  │                                                                          ││
│  └──────────────────────┬──────────────────────────────────────────────────┘│
│                         ▼                                                    │
│  ┌─────────────────────────────────────┐                                    │
│  │ Step 4: MEMORY COMMIT               │                                    │
│  │ memory.add_or_update_feature_       │                                    │
│  │   analysis(file_path, analysis)     │                                    │
│  └─────────────┬───────────────────────┘                                    │
│                │                                                             │
│                ▼                                                             │
│  ┌─────────────────────────────────────┐                                    │
│  │ EXPORT + USAGE REPORT               │                                    │
│  │  ↓                                   │                                    │
│  │ export_to_csv(result, csv_path)     │                                    │
│  │  ↓                                   │                                    │
│  │ usage_tracker.get_session_summary() │                                    │
│  │  ↓                                   │                                    │
│  │ Print: token usage, cost, limits    │                                    │
│  └─────────────────────────────────────┘                                    │
│                                                                              │
│  END                                                                         │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Setup & Installation

### Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | ≥ 3.9 | Tested with 3.10, 3.11, 3.12 |
| pip | Latest | Comes with Python |
| Git | Any | For cloning the repo |
| Azure OpenAI | Active subscription | For full AI mode. Optional for local fallback |

### Step 1: Clone the Repository

```powershell
git clone https://github.com/swapnilchhabilwad/ai_qa_agent.git
cd ai_qa_agent
```

### Step 2: Create and Activate a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate
```

**Windows (cmd):**
```cmd
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```powershell
pip install -r requirements.txt
```

**IMPORTANT:** If you encounter issues with FAISS on Windows, you may need to install it separately:

```powershell
pip install faiss-cpu
# OR for GPU support:
pip install faiss-gpu
```

### Step 4: Create `.env` File

Create a file named `.env` in the project root with this template:

```ini
# ─────────────────────────────────────────────
# AZURE OPENAI CONFIGURATION
# ─────────────────────────────────────────────
AZURE_OPENAI_API_KEY=your_azure_openai_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview

# Deployment names (as configured in Azure AI Studio)
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=gpt-4o          # Chat model deployment
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=text-embedding-3-small  # Embedding model

# ─────────────────────────────────────────────
# TOKEN BUDGET & PRICING
# ─────────────────────────────────────────────
AZURE_OPENAI_TOKEN_LIMIT_DAILY=100000
COST_PER_1M_PROMPT=0.15
COST_PER_1M_COMPLETION=0.60
COST_PER_1M_EMBEDDING=0.10
```

### Step 5: Verify Installation

Run the connectivity test:

```powershell
python scratch_test.py
```

If successful, you'll see the LLM response content and metadata printed.

If you did NOT configure an API key, the framework will run in **local fallback mode** using keyword-based analysis and hardcoded test cases.

---

## 6. Configuration (`.env`)

### Environment Variables Reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `AZURE_OPENAI_API_KEY` | ✅ Yes | — | Your Azure OpenAI API key |
| `AZURE_OPENAI_ENDPOINT` | ✅ Yes | — | Full URL to your Azure OpenAI resource (e.g., `https://my-resource.openai.azure.com/`) |
| `AZURE_OPENAI_CHAT_DEPLOYMENT_NAME` | ✅ Yes | — | Deployment name for the chat model (e.g., `gpt-4o`, `gpt-4o-mini`) |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME` | ✅ Yes | — | Deployment name for the embedding model (e.g., `text-embedding-3-small`) |
| `AZURE_OPENAI_API_VERSION` | ❌ No | `2024-02-15-preview` | API version string |
| `AZURE_OPENAI_TOKEN_LIMIT_DAILY` | ❌ No | `100000` | Daily token budget cap |
| `COST_PER_1M_PROMPT` | ❌ No | `0.15` | Cost per 1M prompt tokens (USD) |
| `COST_PER_1M_COMPLETION` | ❌ No | `0.60` | Cost per 1M completion tokens (USD) |
| `COST_PER_1M_EMBEDDING` | ❌ No | `0.10` | Cost per 1M embedding tokens (USD) |
| `CONFLUENCE_URL` | ❌ No* | — | Confluence base URL (only needed for Confluence pages) |
| `EMAIL` | ❌ No* | — | Email for Confluence API authentication |
| `API_TOKEN` | ❌ No* | — | API token for Confluence |

> \* Required only if using Confluence as a PRD source.

### Placeholder Detection

The framework will refuse to run with placeholder values. If your `.env` contains any of these, it will raise a `RuntimeError`:

- `your_openai_key`, `your_api_key`, `your_azure_endpoint`, `your_azure_deployment`
- `replace_me`, `example`
- Any value starting with `your_`

### Pricing Configuration

The cost calculation uses these defaults (suitable for `gpt-4o-mini` / `text-embedding-3-small`):

```
Prompt tokens:     $0.15 per 1M tokens
Completion tokens: $0.60 per 1M tokens
Embedding tokens:  $0.10 per 1M tokens
```

You can override these in `.env` if you use different models.

---

## 7. How to Run the Project

### Mode 1: CLI (Command Line Interface)

#### Basic Command
```powershell
python main.py --file_path "data/Projects/Project name/Onboarding/prd1.txt"
```

#### All CLI Options
```powershell
python main.py --help
```

Output:
```
usage: main.py [-h] [--query QUERY] [--file_path FILE_PATH] [--project PROJECT]

Generate test cases based on a PRD document.

optional arguments:
  -h, --help            show this help message and exit
  --query QUERY         The query or feature to generate test cases for.
                        (default: "Generate comprehensive test cases exhaustively
                         for all functional capabilities")
  --file_path FILE_PATH The path to the PRD document.
                        (default: "data/Projects/Project name/Onboarding/prd1.txt")
  --project PROJECT     The project name to scope memory and results.
                        (default: "Project name")
```

#### Examples

```powershell
# Basic usage with default project (Project name)
python main.py --file_path "data/Projects/Project name/Onboarding/prd1.txt"

# Specify a different project
python main.py --project orangehrm --file_path "data/Projects/orangehrm/Onboarding/orangehrm onboarding.txt"

# Custom query
python main.py --query "Focus on security and authentication tests" --file_path "data/Projects/Project name/Access_Management/'Disclosure Owner' Role.txt"

# PDF support
python main.py --file_path "data/Projects/Project name/Billing_and_Subscription/plan_details.pdf"
```

### Mode 2: API (FastAPI Web Server)

#### Start the Server
```powershell
python app.py
```

You'll see output like:
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

#### API Endpoints

**1. POST `/generate-tests`** — Run the full test generation pipeline

```powershell
# Using PowerShell
$body = @{
    file_path = "data/Projects/Project name/Onboarding/prd1.txt"
    project = "Project name"
    query = "Generate comprehensive test cases"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/generate-tests" `
  -Method POST `
  -Body $body `
  -ContentType "application/json"
```

```powershell
# Using curl
curl -X POST "http://localhost:8000/generate-tests" ^
  -H "Content-Type: application/json" ^
  -d "{\"file_path\": \"data/Projects/Project name/Onboarding/prd1.txt\", \"project\": \"Project name\", \"query\": \"Generate comprehensive test cases\"}"
```

Example Response:
```json
{
  "status": "success",
  "csv_path": "data/Results/Project name/Onboarding/prd1_test_cases.csv",
  "test_cases": "### Category: ...",
  "usage": {
    "session_tokens": 15420,
    "session_cost_usd": 0.0032,
    "daily_tokens_used": 15420,
    "daily_cost_usd": 0.0032,
    "daily_limit": 100000,
    "remaining_tokens": 84580
  }
}
```

**2. GET `/usage`** — Check current session usage

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/usage"
```

Response:
```json
{
  "session_tokens": 15420,
  "session_cost_usd": 0.0032,
  "daily_tokens_used": 15420,
  "daily_cost_usd": 0.0032,
  "daily_limit": 100000,
  "remaining_tokens": 84580
}
```

#### Interactive API Docs

Open your browser and navigate to:
```
http://localhost:8000/docs
```

This opens Swagger UI where you can:
- See all endpoints with their schemas
- Click "Try it out" to test endpoints directly
- View request/response examples

### Mode 3: Local Fallback (No API Key)

If you do NOT configure `AZURE_OPENAI_API_KEY`, the framework runs in **local fallback mode**:

1. **Requirement Analysis (`build_analysis()`)** — Uses keyword matching to extract features, rules, flows, and edges from the text
2. **Test Generation (`build_test_cases()`)** — Generates **31 hardcoded test cases** customized to the inferred subject of the PRD
3. **Impact Analysis** — Returns empty (no regression detection)
4. **Coverage Evaluation** — Uses rule-based heuristic only (no LLM evaluation)

You will see this message:
```
Runtime mode: local fallback (Generic responses)
```

---

## 8. All Libraries & Dependencies

### Primary Dependencies (`requirements.txt`)

| Library | Version | Purpose |
|---|---|---|
| **`langchain-openai`** | ≥ 0.2.0 | Azure OpenAI chat integration (ChatOpenAI, AzureChatOpenAI, AzureOpenAIEmbeddings) |
| **`langchain-community`** | ≥ 0.3.0 | Callback manager for token tracking, FAISS vector store |
| **`langchain-text-splitters`** | ≥ 0.3.0 | `RecursiveCharacterTextSplitter` for document chunking |
| **`langchain-core`** | ≥ 0.3.9999 | Core LangChain abstractions |
| **`fastapi`** | ≥ 0.115.0 | Web framework for the REST API |
| **`uvicorn`** | ≥ 0.32.0 | ASGI server to serve FastAPI |
| **`pydantic`** | ≥ 2.0 | Request/response schema validation |
| **`faiss-cpu`** | ≥ 1.9.0 | Vector similarity search (CPU version) |
| **`faiss-gpu`** | (optional) | GPU-accelerated vector search |
| **`openai`** | — | Underlying OpenAI/Azure SDK |
| **`tiktoken`** | — | Token counting for cost estimation |
| **`pypdf`** | — | PDF text extraction |
| **`beautifulsoup4`** | — | HTML cleaning (Confluence pages) |
| **`requests`** | — | HTTP requests (Confluence API) |
| **`python-dotenv`** | — | `.env` file loading |

### Full Installation Command

```powershell
pip install langchain-openai langchain-community langchain-text-splitters langchain-core fastapi uvicorn pydantic faiss-cpu openai tiktoken pypdf beautifulsoup4 requests python-dotenv
```

Or simply:
```powershell
pip install -r requirements.txt
```

### Optional Dependencies

| Library | When Needed | Install Command |
|---|---|---|
| `faiss-gpu` | GPU acceleration | `pip install faiss-gpu` |
| `pytest` | Running tests | `pip install pytest` |

---

## 9. Complete Function Reference

### 9.1 `main.py` — CLI Entry Point

| Function | Signature | Purpose |
|---|---|---|
| `get_configured_sources()` | `(file_path: str) -> list[tuple[str, str]]` | Determines file type (txt/pdf) from extension |
| `load_data()` | `(file_path: str) -> tuple[str, dict]` | Loads file content and generates SHA-256 manifest |
| `init_vector_store()` | `(text: str, manifest: dict)` | Initializes FAISS or local vector store |
| `run_agent()` | `(query: str, file_path: str, project_name: str = "default") -> str` | **Main orchestrator** — runs the full pipeline |

### 9.2 `app.py` — FastAPI Server

| Function | Signature | Purpose |
|---|---|---|
| `save_results_logic()` | `(result: str, file_path: str, project_name: str) -> str` | Saves test results to CSV, returns path |
| `generate_tests()` | `POST /generate-tests` | Async handler for test generation |
| `get_usage()` | `GET /usage` | Returns current session usage summary |
| `load_environment()` | Called at module level | Loads .env before server starts |

### 9.3 `agents/analyzer.py` — AI Analysis

| Function | Signature | Purpose |
|---|---|---|
| `_get_llm()` | `() -> AzureChatOpenAI` | Singleton: returns cached AzureChatOpenAI instance |
| `analyze_requirement()` | `(context: str) -> str` | Extracts Features, User Flows, Business Rules, Edge Conditions |
| `analyze_impact()` | `(new_analysis: str, historical_context: str) -> str` | Compares new vs. historical features for regression detection |

### 9.4 `agents/test_generator.py` — Test Generation Engine

**Core Functions:**

| Function | Purpose |
|---|---|
| `generate_test_cases(context, impact_analysis)` | **Main entry point** — runs Pass 1 + Pass 2 self-evaluation loop |
| `_generate_initial_cases(context, impact_analysis)` | Pass 1: Generates tests across all 21 categories with expansion |
| `_generate_initial_cases_with_fallback(context)` | Pass 1 fallback when no LLM available |
| `_generate_gap_cases(context, impact_analysis, existing, report, pass)` | Pass 2: Generates tests for identified coverage gaps |
| `_generate_gap_cases_with_fallback(context, impact, report, pass)` | Pass 2 fallback with template-based gap cases |
| `_evaluate_coverage(context, impact_analysis, test_cases)` | Evaluates coverage using heuristic + optional LLM |
| `_build_heuristic_coverage_report(context, impact, test_cases)` | Rule-based coverage evaluation (always runs) |
| `_build_coverage_evaluation_prompt(context, impact, test_cases)` | Creates LLM prompt for coverage evaluation |
| `_merge_coverage_reports(llm_report, heuristic_report)` | Merges LLM and heuristic reports (pessimistic merge) |
| `_apply_supplemental_quality_gates(report, test_cases)` | Enforces 11 quality gates for minimum coverage |
| `_determine_coverage_verdict(report)` | Returns `FULL_COVERAGE` or `GAPS_FOUND` |
| `_decorate_test_cases(test_cases, default_category, source_pass)` | Assigns categories, deduplicates, infers dimensions |
| `_determine_output_category(test_case, fallback)` | Maps a test case to the correct output category |
| `_infer_dimensions_from_case(test_case)` | Infer coverage dimensions from test case text |
| `_parse_generation_response(response, default_category, pass)` | Parses LLM response → TestCase objects |
| `_extract_json_payload(response)` | Extracts JSON from LLM response (handles markdown fences) |
| `_parse_generation_response(response, default_category, pass)` | Try JSON first, fallback to markdown parser |
| `_build_generation_prompt(context, category, impact, existing)` | Creates Pass 1 generation prompt |
| `_build_generation_expansion_prompt(context, category, impact, existing)` | Creates Pass 1 expansion sweep prompt |
| `_build_gap_generation_prompt(context, impact, existing, report, pass)` | Creates Pass 2 gap-filling prompt |
| `_invoke_llm(prompt, task_label, temperature)` | Generic LLM call with token tracking |
| `_filter_new_cases(candidates, existing)` | Filters out test cases that already exist |
| `_case_catalog(test_cases, include_steps)` | Creates a human-readable catalog of test cases |
| `_case_matches_quality_gate(test_case, gate)` | Checks if a test case matches a quality gate |
| `_make_template_case(...)` | Creates a TestCase from template parameters |
| `_sort_test_cases_for_output(test_cases)` | Sorts by category order, pass, ID |
| `_legacy_generate_test_cases(context, impact)` | Original single-pass generator (preserved for reference) |

**Prompt Builders:**

| Function | Purpose |
|---|---|
| `_build_generation_prompt()` | Pass 1: Generate net-new tests for a slice |
| `_build_generation_expansion_prompt()` | Pass 1: Expand with overlooked/rare test cases |
| `_build_gap_generation_prompt()` | Pass 2: Fill identified coverage gaps |
| `_build_coverage_evaluation_prompt()` | LLM-based coverage evaluation prompt |

### 9.5 `utils/config.py` — Configuration

| Function | Purpose |
|---|---|
| `load_environment()` | Loads `.env` file (cached, runs once) |
| `has_valid_env(name)` | Checks if env var exists and is not a placeholder |
| `has_valid_openai_key()` | Checks `AZURE_OPENAI_API_KEY` validity |
| `get_required_env(name)` | Fetches env var or raises `RuntimeError` |
| `get_chat_model_name()` | Returns `AZURE_OPENAI_CHAT_DEPLOYMENT_NAME` |
| `get_embedding_model_name()` | Returns `AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME` |
| `get_azure_endpoint()` | Returns `AZURE_OPENAI_ENDPOINT` |
| `get_azure_api_version()` | Returns API version (default: `2024-02-15-preview`) |
| `get_daily_token_limit()` | Returns daily token budget (default: 100,000) |
| `get_pricing_rates()` | Returns dict with `prompt`, `completion`, `embedding` rates |

### 9.6 `utils/loader.py` — File Loaders

| Function | Purpose |
|---|---|
| `load_prd(file_path)` | Reads PDF using PyPDF2 |
| `load_txt(file_path)` | Reads plain text with UTF-8 encoding |
| `load_confluence_page(page_id)` | Fetches Confluence page via REST API |
| `clean_html(html)` | Strips HTML tags using BeautifulSoup |
| `_legacy_clean_html_unused(html)` | Older HTML parser (unused, preserved for reference) |
| `_HTMLTextExtractor` class | Primitive HTML-to-text parser |

### 9.7 `utils/chunking.py` — Text Chunking

| Function | Purpose |
|---|---|
| `chunk_text(text)` | Splits text into 1000-char chunks with 200-char overlap |

### 9.8 `utils/vector_store.py` — Vector Search

| Function | Purpose |
|---|---|
| `_get_openai_embeddings()` | Singleton: cached Azure OpenAI Embeddings client |
| `_current_backend()` | Returns `"openai"` or `"local"` |
| `_manifest_payload(source_manifest, backend)` | Builds manifest JSON |
| `_save_manifest(source_manifest, backend)` | Saves manifest to disk |
| `_load_manifest()` | Loads manifest from disk |
| `_create_local_store(chunks, manifest)` | Creates local JSON store for offline mode |
| `create_vector_store(chunks, manifest)` | Creates FAISS index or local store |
| `load_vector_store(manifest)` | Loads FAISS index or local store |
| `vector_store_exists()` | Checks if index files exist |
| `vector_store_matches(manifest)` | Checks if cached index matches current PRD |
| `LocalVectorStore` class | Keyword-based local search engine |
| `LocalDocument` dataclass | Lightweight document for local mode |

### 9.9 `utils/memory.py` — Persistent Memory

| Function | Purpose |
|---|---|
| `MemoryStore.__init__(project_name)` | Initializes per-project memory file |
| `MemoryStore._load()` | Loads JSON from disk with legacy migration |
| `MemoryStore._save()` | Persists memory to disk |
| `MemoryStore.add_or_update_feature_analysis(file_id, text)` | Stores/updates feature analysis |
| `MemoryStore.get_historical_context(exclude_file_id)` | Returns all historical features as formatted string |

### 9.10 `utils/exporter.py` — CSV Export

| Function | Purpose |
|---|---|
| `export_to_csv(markdown_content, output_path)` | Parses markdown → TestCase[] → writes CSV with 9 columns |

### 9.11 `utils/test_suite.py` — Data Models & Utilities

**Data Models (dataclasses):**

| Class | Fields | Purpose |
|---|---|---|
| `TestCase` | case_id, title, preconditions, steps, expected_result, test_type, category, source_pass, coverage_dimensions | Single test case |
| `CoverageCategoryReport` | category, status, reason, evidence_ids, missing_scenarios, untested_risks | Coverage status for one category |
| `CoverageReport` | categories, missing_scenarios, untested_risks, weak_coverage_areas, coverage_score, overall_status, iterations_used | Full coverage evaluation |
| `TestSuiteResult` | final_test_suite, coverage_report, initial_gaps, additional_test_cases | Complete generation output |

**Core Functions:**

| Function | Purpose |
|---|---|
| `normalize_text(value)` | Collapses whitespace, lowercase for comparison |
| `clean_list_item(value)` | Strips markdown list markers (-, *, 1.) |
| `keyword_matches_text(text, keyword)` | Whole-word keyword search with word boundaries |
| `normalize_dimension_name(value)` | Maps arbitrary label to canonical category name |
| `infer_primary_dimension(test_case)` | Infers coverage dimension from test case fields |
| `calculate_coverage_score(reports)` | Computes 0–100 score from category statuses |
| `status_with_icon(status)` | Returns emoji + label (e.g., "✅ Covered") |
| `parse_test_cases(markdown)` | State-machine parser: markdown → TestCase objects |
| `deduplicate_test_cases(cases)` | Removes duplicates via SHA-256 fingerprint |
| `assign_test_case_ids(cases, existing)` | Assigns sequential IDs (PREFIX-001, PREFIX-002, ...) |
| `render_test_case(test_case)` | TestCase → markdown block |
| `render_test_suite(cases)` | TestCase[] → markdown with category headers |
| `render_coverage_report(report)` | CoverageReport → markdown |
| `render_final_report(result)` | TestSuiteResult → final markdown string |

### 9.12 `utils/local_fallback.py` — Offline Fallback

| Function | Purpose |
|---|---|
| `tokenize(text)` | Splits text into lowercase alphanumeric tokens |
| `extract_sentences(text)` | Splits text into unique sentences |
| `_select_sentences(sentences, keywords, pool, limit)` | Keyword-based sentence selection with ranking |
| `infer_subject(text)` | Determines what the document is about |
| `build_analysis(context)` | Keyword-based analysis → Features, Flows, Rules, Edges |
| `_generic_steps(subject, valid)` | Generates test steps for happy/negative paths |
| `_format_case(id, title, steps, expected, type, dimensions)` | Formats a test case block |
| `build_test_cases(analysis)` | Generates 31 hardcoded test cases customized to subject |

### 9.13 `utils/usage_tracker.py` — Billing & Usage

| Function | Purpose |
|---|---|
| `TokenUsageTracker.__init__()` | Initializes session counters, loads daily data |
| `set_current_source(file_path)` | Sets current PRD filename for audit logging |
| `_initialize_history_file()` | Creates billing CSV headers |
| `_load_usage()` | Loads daily usage from JSON log |
| `_save_usage()` | Persists usage to JSON |
| `_log_to_history_csv(task, prompt, completion, total, cost)` | Appends transaction to billing history |
| `_calculate_incremental_cost(prompt, completion, embedding)` | Calculates USD cost for tokens |
| `record_usage(task_label, prompt, completion, total)` | Records chat token usage + cost |
| `record_embedding_usage(token_count, task_label)` | Records embedding token usage + cost |
| `get_session_summary()` | Returns dict: session/daily tokens, costs, limits, remaining |

### 9.14 `scratch_test.py` — LLM Connectivity Test

| Code | Purpose |
|---|---|
| `sys.path.append(...)` | Adds project root to Python path |
| `load_environment()` | Loads .env |
| `_get_llm()` | Initializes LLM |
| `llm.invoke(prompt)` | Sends test prompt to verify connectivity |

---

## 10. Data Structures / Models

### 10.1 `TestCase` Dataclass

```python
@dataclass
class TestCase:
    case_id: str = ""                          # "PHS-001", "FSS-042", etc.
    title: str = ""                            # Short scenario description
    preconditions: list[str] = field(default_factory=list)   # Setup steps
    steps: list[str] = field(default_factory=list)           # Action steps
    expected_result: str = ""                  # Pass/fail criterion
    test_type: str = ""                        # "Happy Path", "Security", etc.
    category: str = "General"                  # One of 21 OUTPUT_TEST_CATEGORIES
    source_pass: str = "Pass 1"                # "Pass 1", "Pass 2 - Iteration 3"
    coverage_dimensions: list[str] = field(default_factory=list)  # ["Happy Path", "UI/UX Validation"]

    def fingerprint(self) -> str:
        """SHA-256 hash of normalized content for deduplication"""
```

### 10.2 `CoverageCategoryReport` Dataclass

```python
@dataclass
class CoverageCategoryReport:
    category: str                             # "Happy Path", "Negative Scenarios", etc.
    status: str                               # "covered" | "partial" | "missing"
    reason: str                               # Explanation for status
    evidence_ids: list[str] = field(default_factory=list)       # Matching test case IDs
    missing_scenarios: list[str] = field(default_factory=list)  # What's still missing
    untested_risks: list[str] = field(default_factory=list)     # Risks from missing coverage
```

### 10.3 `CoverageReport` Dataclass

```python
@dataclass
class CoverageReport:
    categories: list[CoverageCategoryReport]    # One per coverage category
    missing_scenarios: list[str] = field(default_factory=list)
    untested_risks: list[str] = field(default_factory=list)
    weak_coverage_areas: list[str] = field(default_factory=list)
    coverage_score: int = 0                     # 0–100
    overall_status: str = "GAPS_FOUND"          # "FULL_COVERAGE" or "GAPS_FOUND"
    iterations_used: int = 1
```

### 10.4 `TestSuiteResult` Dataclass

```python
@dataclass
class TestSuiteResult:
    final_test_suite: list[TestCase]            # All generated + deduplicated tests
    coverage_report: CoverageReport             # Final coverage evaluation
    initial_gaps: list[str] = field(default_factory=list)      # Gaps after Pass 1
    additional_test_cases: list[TestCase] = field(default_factory=list)  # Pass 2 additions
```

### 10.5 Memory Store Structure

```json
{
  "historical_features": {
    "data/Projects/Project name/Onboarding/prd1.txt": "Features:\n- ...",
    "data/Projects/Project name/Access_Management/'Disclosure Owner' Role.txt": "Features:\n- ..."
  }
}
```

### 10.6 Usage Log Structure

```json
{
  "2026-06-07": {
    "prompt_tokens": 12500,
    "completion_tokens": 3400,
    "embedding_tokens": 2000,
    "total_tokens": 17900,
    "estimated_cost_usd": 0.00345
  }
}
```

---

## 11. Data Flow — Step by Step

### Full Execution Trace

```
INPUT: "data/Projects/Project name/Onboarding/prd1.txt"
│
├─ [1] run_agent("Generate comprehensive...", "prd1.txt", "Project name")
│
├─ [2] load_data("prd1.txt")
│   ├─ get_configured_sources("prd1.txt") → [("txt", "prd1.txt")]
│   ├─ load_txt("prd1.txt") → "Full text of the PRD..."
│   ├─ text_hash = SHA-256(combined_text)
│   └─ Return: (combined_text, {"sources": ["txt:prd1.txt"], "text_hash": "abc..."})
│
├─ [3] init_vector_store(text, manifest)
│   ├─ chunk_text(text) → ["chunk1", "chunk2", ...]  (1000-char each, 200-char overlap)
│   ├─ vector_store_matches(manifest) → False (first run, no cache)
│   ├─ create_vector_store(chunks, manifest)
│   │   ├─ has_valid_openai_key() → True
│   │   ├─ Calculate embedding tokens (tiktoken)
│   │   ├─ FAISS.from_texts(chunks, embeddings) → FAISS index
│   │   ├─ Save index.faiss + index.pkl + manifest.json
│   │   └─ Return: FAISS vector store
│   └─ Return: FAISS vector store
│
├─ [4] db.similarity_search(query, k=20)
│   └─ → [Document1, Document2, ..., Document20] (top 20 most relevant chunks)
│
├─ [5] context = " ".join(doc.page_content for doc in docs)
│
├─ [6] analyze_requirement(context)
│   ├─ has_valid_openai_key() → True
│   ├─ LLM Prompt → extract Features, User Flows, Business Rules, Edge Conditions
│   ├─ Record tokens via usage_tracker.record_usage()
│   └─ Return: "Features:\n- ...\n\nUser Flows:\n- ..."
│
├─ [7] MemoryStore(project_name="Project name")
│   ├─ Opens/Creates "data/Project name_memory.json"
│   └─ Loads historical features
│
├─ [8] memory.get_historical_context(exclude_file_id="prd1.txt")
│   └─ → "Previously Implemented Features:\n--- Source: other_prd.txt ---\n..."
│      (Empty string if this is the first PRD analyzed)
│
├─ [9] analyze_impact(analysis, historical_context)
│   ├─ If no historical context → Return "" (no impact)
│   ├─ If historical exists → LLM Prompt → "NO_IMPACT" or impact report
│   └─ Return: impact analysis text or ""
│
├─ [10] generate_test_cases(analysis, impact_analysis)
│   │
│   ├─ PASS 1: _generate_initial_cases()
│   │   ├─ For each of 21 categories:
│   │   │   ├─ _build_generation_prompt() → LLM prompt
│   │   │   ├─ _invoke_llm() → LLM response (markdown)
│   │   │   ├─ _parse_generation_response() → TestCase[]
│   │   │   ├─ _build_generation_expansion_prompt() → LLM prompt (expansion)
│   │   │   ├─ _invoke_llm() → LLM response (markdown)
│   │   │   └─ _parse_generation_response() → TestCase[] (expansion)
│   │   └─ deduplicate_test_cases(all_cases) → TestCase[]
│   │
│   ├─ PASS 2: Self-Evaluation Loop (max 10 iterations)
│   │   │
│   │   ├─ Iteration 1:
│   │   │   ├─ _evaluate_coverage()
│   │   │   │   ├─ _build_heuristic_coverage_report() → heuristic report
│   │   │   │   ├─ _build_coverage_evaluation_prompt() → LLM call
│   │   │   │   ├─ _normalize_coverage_report_payload() → LLM report
│   │   │   │   └─ _merge_coverage_reports() → merged report
│   │   │   ├─ Score: 85%, Status: GAPS_FOUND
│   │   │   ├─ _generate_gap_cases() → new TestCase[]
│   │   │   └─ Merge into all_test_cases
│   │   │
│   │   ├─ Iteration 2:
│   │   │   ├─ _evaluate_coverage() → Score: 92%, Status: GAPS_FOUND
│   │   │   ├─ _generate_gap_cases() → new TestCase[]
│   │   │   └─ Merge
│   │   │
│   │   ├─ Iteration 3:
│   │   │   ├─ _evaluate_coverage() → Score: 100%, Status: FULL_COVERAGE
│   │   │   └─ BREAK (loop ends)
│   │   │
│   │   └─ Final: all_test_cases → sort → render_final_report()
│   │
│   └─ Return: "## Final Test Suite\n\n### Category: ...\nTest Case ID: ..."
│
├─ [11] memory.add_or_update_feature_analysis("prd1.txt", analysis)
│   └─ Saves analysis to Project name_memory.json for future impact checks
│
├─ [12] EXPORT
│   ├─ Determine output path: "data/Results/Project name/Onboarding/prd1_test_cases.csv"
│   ├─ export_to_csv(result, csv_path)
│   │   ├─ parse_test_cases(result) → TestCase[]
│   │   ├─ deduplicate_test_cases() → TestCase[]
│   │   └─ Write CSV with 9 columns
│   └─ Return: CSV file saved
│
├─ [13] USAGE REPORT
│   ├─ usage_tracker.get_session_summary()
│   └─ Print: session tokens, cost, daily totals, remaining budget
│
└─ OUTPUT: Console shows test cases + CSV saved + usage report
```

---

## 12. Test Suite Documentation

### 12.1 Running Tests

```powershell
# Run all tests
python -m pytest tests/ -v

# Run a specific test file
python -m pytest tests/test_exporter.py -v

# Run a specific test method
python -m pytest tests/test_generator_refinement.py -v -k "test_fallback_generation"

# Run with coverage report (if pytest-cov is installed)
python -m pytest tests/ -v --cov=agents --cov=utils
```

### 12.2 `tests/test_exporter.py` — CSV Export Tests

| Test Method | What It Validates |
|---|---|
| `test_export_to_csv_writes_structured_rows` | Parses 2 test cases from markdown, writes header + 2 data rows, verifies correct values for ID, Generation Source, Coverage Dimensions |

Uses `NonClosingStringIO` to intercept file writes without touching disk.

### 12.3 `tests/test_generator_refinement.py` — Generator Tests

| Test Method | What It Validates |
|---|---|
| `test_fallback_generation_runs_self_evaluation_loop` | In fallback mode (no API key), the generator still runs the full pipeline, produces `## Final Test Suite`, `## Coverage Report`, `## Additional Test Cases Generated`, at least 6 test cases, and resolves all gaps |
| `test_generation_does_not_stop_just_because_score_reaches_85` | Loop continues past 90% score; only stops on `FULL_COVERAGE`. Uses mocked LLM calls to control coverage reports |
| `test_initial_generation_categories_match_fixed_output_taxonomy` | `INITIAL_GENERATION_CATEGORIES` exactly matches `OUTPUT_TEST_CATEGORIES` |
| `test_decorate_test_cases_maps_ad_hoc_categories_back_to_fixed_taxonomy` | Ad-hoc "Coverage Gap Fill" category → mapped to "API contract, CRUD, integration, and service behavior" |
| `test_heuristic_coverage_requires_api_correctness_evidence` | 3 integration/dependency tests are NOT enough; API correctness tests are required for "covered" status |
| `test_heuristic_coverage_requires_true_security_coverage` | Performance + endurance tests are NOT enough; security tests required for Non-Functional "covered" status |

### 12.4 `tests/test_test_suite.py` — Core Utility Tests

| Test Method | What It Validates |
|---|---|
| `test_parse_test_cases_extracts_structured_fields` | Correctly extracts case_id, category, preconditions, steps, coverage_dimensions from markdown |
| `test_deduplicate_test_cases_removes_content_duplicates` | Duplicating a case and running dedup reduces count by 1 |
| `test_coverage_score_uses_partial_credit` | 7 covered (1.0 each) + 2 partial (0.6 each) + 1 missing (0.0) = 8.2/10 = 82% |

---

## 13. Output Formats

### 13.1 Console Output (CLI)

When you run `python main.py`, you'll see:

```
Runtime mode: OpenAI-enabled

Loading TXT PRD: data/Projects/Project name/Onboarding/prd1.txt
Loaded 1 PRD source(s).
Creating embeddings...

Retrieved context ready.


=== ANALYSIS ===

New PRD Analysis Completed.

Starting Pass 1 test generation...
Generating: Foundational Smoke and Sanity...
Generating: End-to-End Business Workflows...
... (all 21 categories)
Generating: Security, Auth, and Role-Based Access Control...

Starting Pass 2 self-evaluation...
Coverage iteration 1: score=80% verdict=GAPS_FOUND
Coverage iteration 2: score=93% verdict=GAPS_FOUND
Coverage iteration 3: score=100% verdict=FULL_COVERAGE

[Memory Updated] Agent safely committed PRD to System Memory.

===== 🧪 GENERATED TEST CASES =====

## Final Test Suite

### Category: Foundational Smoke and Sanity
Test Case ID: FSS-001
...
(All generated test cases)

[Export Success] Successfully exported 147 test cases to: data/Results/Project name/Onboarding/prd1_test_cases.csv

=============================================
📈 AZURE OPENAI USAGE & BILLING REPORT
=============================================
Session Tokens:     22,450
Session Cost:       $0.0042
---------------------------------------------
Daily Total Tokens: 22,450
Daily Total Cost:   $0.0042
Daily Token Limit:  100,000
Remaining (Budget): 77,550
=============================================

Persistent billing log updated at: data/usage_log.json
```

### 13.2 CSV Output

The exported CSV has these columns:

| Column | Example |
|---|---|
| Category | Positive Happy Path Scenarios |
| Test Case ID | PHS-001 |
| Title | Verify successful user registration with valid data |
| Preconditions | User has a valid email address\nUser is on the registration page |
| Steps | 1. Enter valid email\n2. Enter valid password\n3. Submit form |
| Expected Result | User is registered successfully and redirected to dashboard |
| Test Type | Happy Path |
| Generation Source | Pass 1 |
| Coverage Dimensions | Happy Path, UI/UX Validation |

### 13.3 API Response (JSON)

```json
{
  "status": "success",
  "csv_path": "data/Results/Project name/Onboarding/prd1_test_cases.csv",
  "test_cases": "## Final Test Suite\n\n### Category: ...",
  "usage": {
    "session_tokens": 22450,
    "session_cost_usd": 0.0042,
    "daily_tokens_used": 22450,
    "daily_cost_usd": 0.0042,
    "daily_limit": 100000,
    "remaining_tokens": 77550
  }
}
```

### 13.4 Billing History CSV

```
Timestamp,Project,Source,Task,Prompt_Tokens,Completion_Tokens,Total_Tokens,Estimated_Cost_USD
2026-06-07 14:00:00,Project name,prd1.txt,Requirement Analysis,12500,3400,15900,$0.001215
2026-06-07 14:01:00,Project name,prd1.txt,Test Gen: Foundational Smoke and Sanity,8500,2100,10600,$0.000855
...
```

---

## 14. Troubleshooting & Common Issues

### 14.1 "Runtime mode: local fallback (Generic responses)"

**Cause:** `AZURE_OPENAI_API_KEY` is missing, empty, or contains a placeholder value.

**Fix:** Add a valid Azure OpenAI API key to `.env`:
```ini
AZURE_OPENAI_API_KEY=sk-your-real-key-here
```

### 14.2 "Missing required environment variable 'AZURE_OPENAI_API_KEY'"

**Cause:** The `.env` file doesn't exist or is not in the project root.

**Fix:** Create `.env` in the project root with the required variables. Copy from the template in [Section 5.4](#step-4-create-env-file).

### 14.3 "OpenAI analysis unavailable (AuthenticationError)"

**Cause:** The API key is invalid, expired, or the endpoint is wrong.

**Fix:**
1. Check your key: `AZURE_OPENAI_API_KEY` in `.env`
2. Check your endpoint: `AZURE_OPENAI_ENDPOINT`
3. Verify the deployment name: `AZURE_OPENAI_CHAT_DEPLOYMENT_NAME`
4. Check Azure subscription has available quota

### 14.4 "Existing embeddings do not match the current PRD source set"

**Cause:** The manifest hash in `embeddings/manifest.json` doesn't match the current PRD's hash.

**Fix:** Delete the `embeddings/` folder and re-run. The framework will rebuild the index.

```powershell
Remove-Item -Recurse -Force embeddings/
python main.py --file_path "data/Projects/Project name/Onboarding/prd1.txt"
```

### 14.5 "No test cases could be parsed from the output"

**Cause:** The LLM returned malformed markdown that the parser couldn't understand.

**Fix:** This is rare. Check `debug_raw_llm_output.txt` for the raw LLM response. If it happens frequently, try a different model deployment.

### 14.6 "Error: Could not find PRD document at..."

**Cause:** The file path doesn't exist.

**Fix:** Make sure you're using the correct path. Use absolute paths or paths relative to the project root:
```powershell
python main.py --file_path "data/Projects/Project name/Onboarding/prd1.txt"
```

### 14.7 "FAISS not found" or "ModuleNotFoundError: No module named 'faiss'"

**Cause:** FAISS is not installed.

**Fix:**
```powershell
pip install faiss-cpu
```

### 14.8 Token Limit Exceeded

**Cause:** You've hit the `AZURE_OPENAI_TOKEN_LIMIT_DAILY` (default: 100,000 tokens).

**Fix:**
1. Wait until the next day (usage resets daily)
2. Or increase the limit in `.env`:
```ini
AZURE_OPENAI_TOKEN_LIMIT_DAILY=500000
```

### 14.9 Slow Generation

**Cause:** Generating tests for 21 categories with expansion + self-evaluation loops can take 5–15 minutes depending on PRD complexity and API latency.

**Optimization tips:**
- Use a faster model: `gpt-4o-mini` instead of `gpt-4o`
- Reduce `MAX_ITERATIONS` in `agents/test_generator.py` (line 2023)
- Set higher `temperature` for fewer refinement passes (less deterministic = fewer gaps found)

### 14.10 Confluence Connection Failed

**Cause:** Missing or incorrect Confluence credentials.

**Fix:** Add these to `.env`:
```ini
CONFLUENCE_URL=https://your-domain.atlassian.net
EMAIL=your-email@example.com
API_TOKEN=your-confluence-api-token
```

---

## 15. Extending the Framework

### 15.1 Adding a New File Loader

1. Add a new function in `utils/loader.py`:
```python
def load_docx(file_path: str) -> str:
    """Reads a .docx file and returns its text content."""
    # Your implementation here
    return text
```

2. Update `get_configured_sources()` in `main.py`:
```python
elif file_path.endswith(".docx"):
    sources.append(("docx", file_path))
```

3. Update `load_data()` in `main.py`:
```python
elif source_type == "docx":
    print(f"Loading DOCX PRD: {source_value}")
    text = load_docx(source_value)
    label = f"docx:{source_value}"
```

### 15.2 Adding a New Test Category

1. Add the category name to `OUTPUT_TEST_CATEGORIES` in `agents/test_generator.py`:
```python
OUTPUT_TEST_CATEGORIES = [
    ...
    "Your New Category",
]
```

2. Add slice guidance in `CATEGORY_SLICE_GUIDANCE`:
```python
CATEGORY_SLICE_GUIDANCE = {
    ...
    "Your New Category": "Focus on ...",
}
```

3. Add to `COVERAGE_CATEGORIES` in `utils/test_suite.py`:
```python
COVERAGE_CATEGORIES = [
    ...
    "Your New Category",
]
```

4. Add a prefix in `CATEGORY_PREFIXES`:
```python
CATEGORY_PREFIXES = {
    ...
    "Your New Category": "YNC",
}
```

5. Add keywords in `CATEGORY_KEYWORDS`:
```python
CATEGORY_KEYWORDS = {
    ...
    "Your New Category": ("keyword1", "keyword2"),
}
```

### 15.3 Adding a Quality Gate

Add a new entry to `SUPPLEMENTAL_QUALITY_GATES` in `agents/test_generator.py`:

```python
SUPPLEMENTAL_QUALITY_GATES = [
    ...
    {
        "name": "Your quality gate description",
        "owner": "Happy Path",
        "output_categories": ["Positive Happy Path Scenarios"],
        "keywords": ("keyword1", "keyword2"),
        "min_cases": 2,
        "missing_scenario": "Add coverage for ...",
        "untested_risk": "Risk description if not covered",
    },
]
```

### 15.4 Using a Different LLM Provider

Replace the LLM initialization in `_get_llm()` functions:

1. In `agents/analyzer.py` (line 19–41):
```python
def _get_llm():
    from langchain_community.chat_models import ChatOpenAI  # Standard OpenAI
    return ChatOpenAI(
        model="gpt-4",
        api_key=get_required_env("OPENAI_API_KEY"),
        temperature=0,
    )
```

2. In `agents/test_generator.py` (line 2031–2047): Same approach.

---

## 16. Glossary

| Term | Definition |
|---|---|
| **PRD** | Product Requirement Document — the input file describing features |
| **RAG** | Retrieval-Augmented Generation — retrieves relevant context before generating |
| **FAISS** | Facebook AI Similarity Search — library for efficient vector similarity search |
| **Vector Store** | Database of embeddings for semantic search |
| **Chunking** | Splitting large text into smaller segments for LLM context windows |
| **Embedding** | Numerical vector representation of text for similarity comparison |
| **Manifest** | JSON metadata mapping source hashes to cached vector indices |
| **Pass 1** | Initial test generation across all 21 categories |
| **Pass 2** | Self-evaluation and gap-filling refinement loop |
| **Coverage Score** | 0–100 metric based on status of all 10 coverage dimensions |
| **FULL_COVERAGE** | All coverage categories are marked "covered" |
| **GAPS_FOUND** | At least one coverage category is "partial" or "missing" |
| **Quality Gates** | 11 minimum-coverage rules enforced by the heuristic evaluator |
| **MemoryStore** | Persistent JSON store for cross-PRD feature history |
| **Impact Analysis** | Regression detection by comparing new vs. historical features |
| **Local Fallback** | Offline mode using keyword matching + hardcoded templates |
| **Usage Tracker** | Real-time token/cost monitoring with daily budget enforcement |
| **Slice Guidance** | Per-category instructions telling the LLM what to focus on |
| **Coverage Dimensions** | The 10 QA coverage areas test cases are evaluated against |
| **LLM Runtime Unavailable** | Flag set after first LLM failure to avoid repeated failed calls |

---

## Quick Reference Card

```powershell
# ─────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────
git clone https://github.com/swapnilchhabilwad/ai_qa_agent.git
cd ai_qa_agent
python -m venv venv
.\venv\Scripts\Activate
pip install -r requirements.txt
# Create .env with your Azure OpenAI credentials

# ─────────────────────────────────────────
# RUN CLI
# ─────────────────────────────────────────
python main.py --project Project name --file_path "data/Projects/Project name/Onboarding/prd1.txt"
python main.py --project orangehrm --file_path "data/Projects/orangehrm/Onboarding/orangehrm onboarding.txt"
python main.py --help

# ─────────────────────────────────────────
# RUN API
# ─────────────────────────────────────────
python app.py
# Open http://localhost:8000/docs in browser

# ─────────────────────────────────────────
# RUN TESTS
# ─────────────────────────────────────────
python -m pytest tests/ -v

# ─────────────────────────────────────────
# CONNECTIVITY TEST
# ─────────────────────────────────────────
python scratch_test.py

# ─────────────────────────────────────────
# CLEAR CACHE
# ─────────────────────────────────────────
Remove-Item -Recurse -Force embeddings/
