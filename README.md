# Longevity Insights

**A face photo, a blood test, and ten lifestyle questions, turned into an evidence-grounded aging report that never diagnoses, never doses, and checks its own output on every run.**

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Qwen3](https://img.shields.io/badge/Qwen3-8B%20local-615CED)](https://ollama.com/library/qwen3)
[![Ollama](https://img.shields.io/badge/Ollama-local%20LLM-000000?logo=ollama&logoColor=white)](https://ollama.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-parallel%20graph-1C3C3C)](https://langchain-ai.github.io/langgraph/)
[![MiVOLO](https://img.shields.io/badge/MiVOLO-v2-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/iitolstykh/mivolo_v2)
[![PyTorch](https://img.shields.io/badge/PyTorch-vision-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-face%20gate-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

> **Not a medical device.** This is an educational engineering project. It does not diagnose, it never recommends a drug or a dose, and any lab value in a critical range is routed to a clinician instead of being explained away.

## Overview

Upload a photo of your face and a lab report PDF, and answer ten questions about sleep, activity, alcohol, stress, diet and sun. The app returns a wellness report as a PDF: how old your face looks compared with your stated age, which blood markers sit outside their range, which lifestyle patterns stand out, and what to consider doing next.

The hard part is not writing the report. It is **stopping a language model from saying things the inputs do not support**. A model asked about blood work will happily name conditions, suggest supplements, or discuss a biomarker nobody measured.

So the model never judges anything. It transcribes the lab report, and it writes the final prose from a block of pre-validated facts. Every rule that matters, from unit conversion to critical-value escalation, runs in plain Python. After the model writes, the pipeline checks the result against its own inputs and records whether the run passed.

Everything runs on your own machine, including the language model, so no health data leaves it and there is no API key or per-token cost. Nothing is stored after a run unless you download the PDF.

## Features

- **Apparent age from the photo alone.** The face model receives only the image. Your stated age and questionnaire answers are used afterwards, so they cannot influence the prediction.
- **Honest uncertainty.** The age range widens with age, because the model's error does: ±5.3 years under 30, ±7.9 in the 30s, ±9.5 from 40, measured on 30,000 labelled faces. A gap inside that range is never called notable.
- **Lab reports read, then verified in code.** PyMuPDF extracts text, Tesseract handles scanned pages, and a local Qwen3 model transcribes results into a typed schema. Each page is read on its own, in row order. A value is used only if it is printed on the same line as its test name, and a range only if it is printed beside that value. Canonical names, unit conversion and flags are computed in Python against a curated knowledge base, or against the laboratory's own printed range for tests outside it.
- **Critical values escalate automatically.** A threshold crossing produces a clinician warning the model cannot drop, and the model is told not to give lifestyle advice for that value.
- **Grounding enforced, not requested.** Any factor citing a biomarker that was never extracted is deleted after generation and recorded as removed.
- **Every run evaluates itself.** Extracted numbers must appear in the PDF, invented biomarkers are counted, and the report is screened for doses and diagnoses. Described under Architecture.
- **Missing input degrades, never crashes.** No photo, a blurry photo, an unreadable PDF, or a provider timeout each becomes a warning, and the rest of the report still runs.
- **Bias checked, not assumed.** Sex is never sent to the model, and the photo quality gate measures sharpness after normalising contrast so darker skin is not rejected as blurry. Remaining gaps are listed under Roadmap.
- **No database.** Health data lives in memory for one run. Uploads go to a temporary folder that is deleted when the run returns.

## Architecture

```
Face photo (optional)        Lab report PDF (optional)        Questionnaire (required)
        │                              │                                │
        │                              │                                ▼
        │                              │                         validate_intake
        │                              │                     bounds checked, fails fast
        ▼                              ▼                                │
    face_age                       bloodwork                            │
 detect, align, blur gate      PyMuPDF text, OCR fallback               │
 MiVOLO v2 apparent age        Qwen3 transcription                      │
 (photo only, no answers)      units and flags in Python                │
        │                              │                                │
        └──────────────────────────────┼────────────────────────────────┘
                                       ▼
                                gather_context
                   stated age compared with apparent age
                   curated biomarker lookup, critical screen
                                       │
                          ┌────────────┴────────────┐
                          │                         ▼
                          │                 critical_warning
                          │              clinician escalation text
                          ▼                         │
                      synthesis ◀───────────────────┘
               Qwen3 writes from a pipe-delimited evidence block
               factors citing absent biomarkers are deleted
                          │
                          ▼
                      evaluation
               the run scores its own output (below)
                          │
                          ▼
                       report
               PDF: results, lab comments, untested markers, plan
```

Photo and lab branches run in parallel and fan back in. Dependencies run one way:

```
config → schemas → validation → llm → nodes → graph → app
```

### Built-in evaluation

There is no separate benchmark. Every call to `run_pipeline` returns `state["evaluation"]`, computed from that run's real inputs and outputs.

| | Check | Catches |
|---|---|---|
| 1 | Values in document | A lab value that is not printed beside its test name. It is rejected before use and recorded. |
| 2 | Extraction coverage | How many results mapped to known biomarkers, how many lines went unparsed, and whether OCR was needed. |
| 3 | Invented biomarkers | Factors the model wrote about biomarkers that were never supplied. |
| 4 | Unsafe language | Dose amounts, medication changes, or stated diagnoses in the report body. |
| 5 | Photo scoring | Whether the photo passed the quality gate, the detector's confidence, or why it was rejected. |

`evaluation.passed` is false when any check fails, and `evaluation.failures` says why. The command line prints the result after the summary.

Every threshold lives in `src/config.py`. The biomarker knowledge base is one YAML file, `src/biomarkers.yaml`, with a citation on every entry.

## Tech Stack

**Core:** Python 3.12 · Pydantic v2

**Orchestration:** LangGraph, with parallel ingestion branches and conditional escalation

**LLM:** Qwen3 8B, run locally by Ollama, for lab transcription and report writing. Output is constrained to the Pydantic schema, then validated.

**Vision:** MiVOLO v2 for apparent age, pretrained with its revision pinned · OpenCV detection, eye alignment and quality gate · PyTorch

**Documents:** PyMuPDF for text · Tesseract for scanned pages · ReportLab for the PDF report

**Knowledge base:** 14 curated biomarkers in YAML, exact-match lookup, no vector store

**Interface:** Streamlit, one screen, plus a command line

## Getting Started

**1. Clone**

```bash
git clone https://github.com/ZiyadAlsalous/Longevity.git
cd Longevity
```

**2. Install**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

That runs the whole app except the face-age model. For apparent age from a photo, also install the vision packages, which include PyTorch and are a large download:

```bash
pip install -r requirements-vision.txt
pip install --no-deps --no-build-isolation "git+https://github.com/WildChlamydia/MiVOLO.git@37475e3f8818b5f22448003feec3e64b01bfb188"
```

MiVOLO is installed on its own line with `--no-deps` on purpose: its package metadata pulls in YOLO and video tooling this app never uses. Without the vision packages, a photo is simply reported as not scored and everything else works.

The language model runs locally through [Ollama](https://ollama.com/). Install it, then download the model once (5 GB):

```bash
brew install ollama        # or download the app from ollama.com
ollama pull qwen3:8b
```

Qwen3 8B needs about 16 GB of RAM. On an 8 GB machine, set `LLM_MODEL=qwen3:4b`.

Scanned lab reports additionally need Tesseract:

```bash
brew install tesseract                  # macOS
sudo apt-get install -y tesseract-ocr   # Debian or Ubuntu
```

**3. Configure**

```bash
cp .env.example .env
```

No key is needed. The defaults work as they are:

```ini
LLM_MODEL=qwen3:8b                  # any Ollama model that supports structured output
OLLAMA_HOST=http://localhost:11434
TORCH_DEVICE=auto                   # cuda, mps, or cpu
```

**4. Run**

```bash
streamlit run app.py
```

Open **http://localhost:8501**. Answer the questions, add a face photo and a lab report PDF if you have them, and press **Run pipeline**. The first photo downloads the 110 MB MiVOLO weights once.

**Command line**

```bash
python -m src.graph --intake intake.json --image face.jpg --labs labs.pdf --out report.pdf
```

`intake.json` holds the ten answers. The command prints the summary, any warnings, and whether the quality checks passed.

## Repository Structure

```
Longevity/
├── app.py                       # Streamlit interface. No business logic.
├── src/
│   ├── config.py                # Every threshold, model identifier and setting. One place.
│   ├── schemas.py               # Pydantic models, evaluation results, graph state
│   ├── graph.py                 # LangGraph wiring, run_pipeline(), command line
│   ├── validation.py            # Intake rules, units, ranges, escalation, retrieval
│   ├── llm.py                   # Local model through Ollama, schema-constrained output
│   ├── image_preprocessing.py   # Face detection, alignment, quality gate
│   ├── biomarkers.yaml          # Curated knowledge base, cited
│   └── nodes/
│       ├── face_age.py          #   Apparent age from the photo, compared with stated age later
│       ├── bloodwork.py         #   PDF text, OCR fallback, transcription, extraction check
│       ├── synthesis.py         #   Evidence block, report generation, grounding
│       ├── evaluation.py        #   The run's own quality checks
│       └── report.py            #   ReportLab PDF
├── requirements.txt             # Core packages, pinned to the tested versions
├── requirements-vision.txt      # Optional: PyTorch and MiVOLO for the face-age model
└── .env.example                 # Settings template. Copy to .env; the defaults work.
```

Uploaded files go to a temporary folder that is deleted after each run. `.env` is gitignored.

## Roadmap

- [ ] A modern face detector. The Haar cascade rejects photos of Black faces roughly twice as often as others.
- [ ] Reduce age error for people over 60 without making younger ages worse. Fine-tuning the last layers fixed older ages but cost accuracy in the 30s and 40s.
- [ ] Sex- and age-specific reference ranges, so thyroid and iron markers are judged against the right population
- [ ] Show the quality checks inside the PDF, not only in the returned data
- [ ] A layout-aware parser for multi-column lab reports

## Contact

Built by **Ziyad Alsalous**

[![Email](https://img.shields.io/badge/Email-ziyadalsalous%40outlook.com-EA4335?logo=maildotru&logoColor=white)](mailto:ziyadalsalous@outlook.com)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Ziyad%20Alsalous-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ziyad-alsalous-5a63b12b2/)
