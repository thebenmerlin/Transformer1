#!/usr/bin/env python3
"""Practical 3: observable prompt structures and DSPy on local Ollama.

All inference is performed by qwen3:4b at the local Ollama endpoint.  The
GitHub Actions workflow starts that endpoint on an ephemeral runner.
"""

import csv
import importlib.metadata
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import dspy
import ollama


STUDENT_NAME = "Gajanan Barve"
STUDENT_ID = "RBT23AR001"
MODEL = "qwen3:4b"
OLLAMA_HOST = "http://127.0.0.1:11434"
RESULTS_FILE = Path("practical3_results.csv")
TEX_FILE = Path("practical3.tex")

# `think: False` requests Qwen's non-thinking mode. Prompts ask only for
# concise, observable rationale summaries rather than hidden reasoning.
GENERATION_OPTIONS = {"temperature": 0, "num_predict": 180, "think": False}


class BusinessClassifier(dspy.Signature):
    """Classify a company using revenue, employees and industry."""

    company_description: str = dspy.InputField()
    business_category: str = dspy.OutputField(
        desc="A concise illustrative business-size category only."
    )


class ExtractInfo(dspy.Signature):
    """Extract the stated revenue, employee count, and industry."""

    company_text: str = dspy.InputField()
    revenue: str = dspy.OutputField(desc="Revenue exactly as stated in the input.")
    employees: str = dspy.OutputField(desc="Employee count exactly as stated in the input.")
    industry: str = dspy.OutputField(desc="Industry exactly as stated in the input.")


class BusinessCategory(dspy.Signature):
    """Classify a business from extracted company fields."""

    revenue: str = dspy.InputField()
    employees: str = dspy.InputField()
    industry: str = dspy.InputField()
    category: str = dspy.OutputField(
        desc="A concise illustrative business-size category only."
    )


class MultiStepBusinessPipeline(dspy.Module):
    """Two genuine DSPy Predictor stages: extraction then classification."""

    def __init__(self):
        super().__init__()
        self.extract = dspy.Predict(ExtractInfo)
        self.classify = dspy.Predict(BusinessCategory)

    def forward(self, company_text):
        extracted = self.extract(company_text=company_text)
        classified = self.classify(
            revenue=extracted.revenue,
            employees=extracted.employees,
            industry=extracted.industry,
        )
        return dspy.Prediction(
            revenue=extracted.revenue,
            employees=extracted.employees,
            industry=extracted.industry,
            category=classified.category,
        )


def package_version(name):
    return importlib.metadata.version(name)


def shell_output(command):
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return completed.stdout.strip() or "(no output)"


def visible_text(text):
    """Remove any accidental hidden-thinking tags before printing or saving."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    return text.strip()


def get_chat_content(response):
    if isinstance(response, dict):
        return response.get("message", {}).get("content", "")
    return response.message.content


def direct_ollama_call(client, prompt):
    started = time.perf_counter()
    response = client.chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are assisting with an educational prompt-engineering lab. "
                    "Business categories are illustrative, not legal or regulatory. "
                    "Never reveal private chain-of-thought; provide only concise "
                    "observable summaries in the requested format."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        options=GENERATION_OPTIONS,
    )
    elapsed = time.perf_counter() - started
    return visible_text(get_chat_content(response)), elapsed


def extract_labeled_value(text, label):
    match = re.search(rf"(?im)^\s*{re.escape(label)}\s*:\s*(.+?)\s*$", text)
    if match:
        return match.group(1).strip(" `*#")
    return "Label not parsed; see generated response"


def print_environment():
    print("=== Environment ===")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"Operating system: {platform.platform()}")
    print(f"CPU architecture: {platform.machine()}")
    print(f"Ollama Python package version: {package_version('ollama')}")
    print(f"DSPy version: {dspy.__version__}")
    print(f"Ollama version: {shell_output(['ollama', '--version'])}")
    print(f"Model used: {MODEL}")
    print("Runtime: Ollama local endpoint on GitHub-hosted Linux runner")
    print()


def run_part_a(client):
    cot_scenario = "Company: Annual Revenue = $5 Million; Employees = 150; Sector = Healthcare."
    cot_prompt = f"""Classify this illustrative business-size scenario: {cot_scenario}

Return exactly these visible sections. Do not provide hidden chain-of-thought.
Factors Considered:
- factor 1
- factor 2
Short Rationale: one or two concise sentences.
Final Classification: one concise category.
"""
    print("=== Part A: CoT ===")
    print(f"Input Scenario: {cot_scenario}")
    print("Exact Prompt:")
    print(cot_prompt.strip())
    cot_output, cot_time = direct_ollama_call(client, cot_prompt)
    cot_classification = extract_labeled_value(cot_output, "Final Classification")
    print("Generated Output:")
    print(cot_output)
    print(f"Execution Time: {cot_time:.3f} s\n")

    tot_scenario = "Revenue = $2 Million; Employees = 80; Sector = Retail."
    tot_prompt = f"""For this illustrative business-size scenario: {tot_scenario}

Propose and compare three candidates. Return exactly these visible labels and
keep the evaluation concise. Do not provide hidden chain-of-thought.
Candidate 1: category - brief consideration
Candidate 2: category - brief consideration
Candidate 3: category - brief consideration
Brief Evaluation: concise comparison of the candidates.
Selected Classification: one concise category.
"""
    print("=== Part A: ToT ===")
    print(f"Input Scenario: {tot_scenario}")
    print("Exact Prompt:")
    print(tot_prompt.strip())
    tot_output, tot_time = direct_ollama_call(client, tot_prompt)
    tot_classification = extract_labeled_value(tot_output, "Selected Classification")
    print("Generated Output:")
    print(tot_output)
    print(f"Execution Time: {tot_time:.3f} s\n")

    react_scenario = "Employees = 500; Annual Revenue = $50 Million."
    react_prompt = f"""For this illustrative business-size scenario: {react_scenario}

Use only this observable ReAct-style format. The action is an internal
characteristics evaluation, not a web or tool call. Do not provide hidden
chain-of-thought.
Reasoning Summary: brief summary of relevant factors.
Action: evaluate company scale using the stated employees and revenue.
Observation: concise outcome of that evaluation.
Final Answer: one concise business-size category.
"""
    print("=== Part A: ReAct ===")
    print(f"Input Scenario: {react_scenario}")
    print("Exact Prompt:")
    print(react_prompt.strip())
    react_output, react_time = direct_ollama_call(client, react_prompt)
    react_classification = extract_labeled_value(react_output, "Final Answer")
    print("Generated Output:")
    print(react_output)
    print(f"Execution Time: {react_time:.3f} s\n")

    return {
        "CoT-style": {
            "scenario": cot_scenario,
            "output": cot_output,
            "classification": cot_classification,
            "time": cot_time,
            "calls": 1,
        },
        "ToT-style": {
            "scenario": tot_scenario,
            "output": tot_output,
            "classification": tot_classification,
            "time": tot_time,
            "calls": 1,
        },
        "ReAct-style": {
            "scenario": react_scenario,
            "output": react_output,
            "classification": react_classification,
            "time": react_time,
            "calls": 1,
        },
    }


def configure_dspy():
    # Current DSPy documented local Ollama adapter; api_key is intentionally empty.
    lm = dspy.LM(
        f"ollama_chat/{MODEL}",
        api_base=OLLAMA_HOST,
        api_key="",
        model_type="chat",
        temperature=0,
        max_tokens=120,
        cache=False,
        think=False,
    )
    dspy.configure(lm=lm)
    return lm


def run_part_b():
    print("=== Part B: DSPy Predictor ===")
    lm = configure_dspy()
    print(f"DSPy LM backend: ollama_chat/{MODEL} at {OLLAMA_HOST}")
    predictor_input = "Company Revenue: $3 Million\nEmployees: 120\nIndustry: Retail"
    predictor = dspy.Predict(BusinessClassifier)
    started = time.perf_counter()
    predictor_result = predictor(company_description=predictor_input)
    predictor_time = time.perf_counter() - started
    predictor_category = str(predictor_result.business_category).strip()
    print("Input:")
    print(predictor_input)
    print(f"DSPy Business Category: {predictor_category}")
    print(f"Execution Time: {predictor_time:.3f} s\n")

    print("=== Part B: Multi-Step DSPy Pipeline ===")
    pipeline_input = "ABC Retail Ltd.\nAnnual Revenue: $4 Million\nEmployees: 180\nIndustry: Retail"
    pipeline = MultiStepBusinessPipeline()
    started = time.perf_counter()
    pipeline_result = pipeline(company_text=pipeline_input)
    pipeline_time = time.perf_counter() - started
    revenue = str(pipeline_result.revenue).strip()
    employees = str(pipeline_result.employees).strip()
    industry = str(pipeline_result.industry).strip()
    pipeline_category = str(pipeline_result.category).strip()
    print("Input:")
    print(pipeline_input)
    print(f"Revenue: {revenue}")
    print(f"Employees: {employees}")
    print(f"Industry: {industry}")
    print(f"Business Category: {pipeline_category}")
    print(f"Execution Time: {pipeline_time:.3f} s\n")

    return {
        "DSPy Predictor": {
            "scenario": predictor_input.replace("\n", "; "),
            "output": f"DSPy Business Category: {predictor_category}",
            "classification": predictor_category,
            "time": predictor_time,
            "calls": 1,
        },
        "DSPy Multi-Step Pipeline": {
            "scenario": pipeline_input.replace("\n", "; "),
            "output": (
                f"Revenue: {revenue}\nEmployees: {employees}\nIndustry: {industry}\n"
                f"Business Category: {pipeline_category}"
            ),
            "classification": pipeline_category,
            "time": pipeline_time,
            "calls": 2,
        },
        "dspy_lm": lm,
    }


def write_results(results):
    fields = [
        "technique",
        "scenario",
        "model",
        "classification",
        "llm_calls",
        "execution_time_seconds",
        "notes",
    ]
    with RESULTS_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for technique, item in results.items():
            if technique == "dspy_lm":
                continue
            writer.writerow(
                {
                    "technique": technique,
                    "scenario": item["scenario"],
                    "model": MODEL,
                    "classification": item["classification"],
                    "llm_calls": item["calls"],
                    "execution_time_seconds": f"{item['time']:.6f}",
                    "notes": "Illustrative business-size classification; complete output is in practical3_output.txt.",
                }
            )


def tex_escape(value):
    substitutions = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(substitutions.get(character, character) for character in str(value))


def listing_text(value, max_characters=900):
    value = str(value).strip()
    if len(value) > max_characters:
        value = value[:max_characters].rsplit("\n", 1)[0] + "\n[Output excerpt truncated; full output is in practical3_output.txt.]"
    return value.encode("ascii", "replace").decode("ascii")


def build_latex(results):
    cot = results["CoT-style"]
    tot = results["ToT-style"]
    react = results["ReAct-style"]
    predictor = results["DSPy Predictor"]
    pipeline = results["DSPy Multi-Step Pipeline"]
    table_rows = "\n".join(
        f"{tex_escape(name)} & {tex_escape(item['classification'])} & {item['calls']} & {item['time']:.3f} \\\\"
        for name, item in results.items()
        if name != "dspy_lm"
    )
    replacements = {
        "@@DSPY_VERSION@@": tex_escape(dspy.__version__),
        "@@OLLAMA_VERSION@@": tex_escape(shell_output(["ollama", "--version"])),
        "@@COT_OUTPUT@@": listing_text(cot["output"]),
        "@@TOT_OUTPUT@@": listing_text(tot["output"]),
        "@@REACT_OUTPUT@@": listing_text(react["output"]),
        "@@PREDICTOR_OUTPUT@@": listing_text(predictor["output"]),
        "@@PIPELINE_OUTPUT@@": listing_text(pipeline["output"]),
        "@@TABLE_ROWS@@": table_rows,
        "@@COT_CLASS@@": tex_escape(cot["classification"]),
        "@@TOT_CLASS@@": tex_escape(tot["classification"]),
        "@@REACT_CLASS@@": tex_escape(react["classification"]),
        "@@PREDICTOR_CLASS@@": tex_escape(predictor["classification"]),
        "@@PIPELINE_CLASS@@": tex_escape(pipeline["classification"]),
    }
    template = r"""\documentclass[10pt]{article}
\usepackage[a4paper,margin=0.67in,headheight=15pt,headsep=15pt,footskip=24pt]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{newtxtext,newtxmath}
\usepackage{microtype,booktabs,tabularx,array,enumitem,xcolor,listings,fancyhdr,titlesec,amsmath,caption,tikz,hyperref}
\usetikzlibrary{positioning,arrows.meta}
\definecolor{accent}{HTML}{17365D}
\definecolor{shade}{HTML}{F3F6F9}
\hypersetup{colorlinks=true,linkcolor=accent,urlcolor=accent}
\setlength{\parindent}{0pt}
\setlength{\parskip}{3.2pt}
\setlist[itemize]{leftmargin=15pt,itemsep=1pt,topsep=2pt}
\titleformat{\section}{\color{accent}\normalfont\bfseries\large}{\thesection.}{0.45em}{}
\titlespacing*{\section}{0pt}{7pt}{3pt}
\titleformat{\subsection}{\color{accent}\normalfont\bfseries\normalsize}{\thesubsection}{0.4em}{}
\titlespacing*{\subsection}{0pt}{5pt}{2pt}
\renewcommand{\arraystretch}{1.12}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small\sffamily Generative AI Laboratory}
\fancyhead[R]{\small\sffamily Practical 3}
\fancyfoot[L]{\small Gajanan Barve}
\fancyfoot[C]{\small RBT23AR001}
\fancyfoot[R]{\small\thepage}
\renewcommand{\headrulewidth}{0.35pt}
\renewcommand{\footrulewidth}{0.35pt}
\lstdefinestyle{output}{basicstyle=\ttfamily\scriptsize,backgroundcolor=\color{shade},frame=single,rulecolor=\color{accent!45},framesep=5pt,breaklines=true,columns=fullflexible,aboveskip=4pt,belowskip=4pt}
\lstdefinestyle{code}{basicstyle=\ttfamily\scriptsize,backgroundcolor=\color{shade},frame=single,rulecolor=\color{accent!45},framesep=5pt,breaklines=true,columns=fullflexible,aboveskip=3pt,belowskip=4pt}
\begin{document}
\begin{center}
{\small\color{accent}\sffamily GENERATIVE AI LABORATORY}\[-2pt]
{\color{accent}\rule{\linewidth}{0.65pt}}\\[-1pt]
{\large\bfseries PRACTICAL NO. 3}\\[2pt]
{\LARGE\bfseries Declarative Prompt Programming using CoT, ToT, ReAct and DSPy}\\[4pt]
{\normalsize Gajanan Barve \quad | \quad RBT23AR001}
\end{center}

\section{Aim}
To implement observable Chain-of-Thought-style, Tree-of-Thought-style, and ReAct-style prompting, and to develop a two-stage DSPy business-classification pipeline backed by a local open-weight model.

\section{Objectives}
\begin{itemize}
\item Compare three prompt structures for illustrative business-size classification.
\item Configure DSPy Signatures and Predictors over a local Ollama endpoint.
\item Measure genuine end-to-end execution time and record generated classifications.
\end{itemize}

\section{Experimental Setup}
The experiment ran on a GitHub-hosted Linux runner. The runtime was Ollama with \texttt{qwen3:4b} (Qwen3 4B); Ollama reported \texttt{@@OLLAMA_VERSION@@}. DSPy version was \texttt{@@DSPY_VERSION@@}. All categories are illustrative model outputs, not legal or regulatory SME determinations.

\section{Theory}
\textbf{CoT-style prompting} asks for explicit, concise factors and a visible rationale. \textbf{Tree-of-Thought} compares alternatives before selecting one. \textbf{ReAct-style prompting} separates an observable summary, action, observation, and answer. \textbf{DSPy} expresses LLM tasks declaratively through Signatures and modules; Predictors instantiate those input/output contracts over a configured language model.

\section{Methodology and Implementation}
The three Part A prompts constrained output to lab-visible summaries and specifically excluded hidden reasoning. The same local model was configured for DSPy using the documented native Ollama adapter.
\begin{lstlisting}[style=code]
lm = dspy.LM("ollama_chat/qwen3:4b",
             api_base="http://127.0.0.1:11434",
             api_key="", model_type="chat",
             temperature=0, cache=False, think=False)
dspy.configure(lm=lm)

class BusinessClassifier(dspy.Signature):
    company_description: str = dspy.InputField()
    business_category: str = dspy.OutputField()
\end{lstlisting}
\newpage

\section{Experimental Output}
\textbf{CoT-style scenario:} Revenue \$5 Million; 150 employees; Healthcare. The generated visible response was:
\begin{lstlisting}[style=output]
@@COT_OUTPUT@@
\end{lstlisting}
\textbf{ToT-style scenario:} Revenue \$2 Million; 80 employees; Retail. The generated candidate comparison was:
\begin{lstlisting}[style=output]
@@TOT_OUTPUT@@
\end{lstlisting}
\textbf{ReAct-style scenario:} 500 employees; annual revenue \$50 Million. The generated observable structure was:
\begin{lstlisting}[style=output]
@@REACT_OUTPUT@@
\end{lstlisting}

\section{Comparative Analysis}
Table~\ref{tab:comparison} reports actual end-to-end times measured with \texttt{time.perf\_counter()}. The DSPy multi-step pipeline makes two sequential local LM calls; all other rows make one.
\begin{table}[h]
\centering\small
\caption{Measured comparison of prompt and DSPy approaches}\label{tab:comparison}
\begin{tabularx}{\linewidth}{>{\raggedright\arraybackslash}X >{\raggedright\arraybackslash}X c r}
\toprule
Technique & Output / classification & Calls & Time (s)\\
\midrule
@@TABLE_ROWS@@
\bottomrule
\end{tabularx}
\end{table}
\newpage

\section{DSPy Pipeline}
The pipeline first extracts structured facts from the company text and then supplies those fields to a separate classification Predictor.
\begin{center}
\begin{tikzpicture}[node distance=8mm and 17mm,>=Latex,font=\small,
box/.style={draw=accent,rounded corners=2pt,fill=accent!7,minimum width=33mm,minimum height=8mm,align=center},
field/.style={draw=black!55,rounded corners=2pt,fill=shade,minimum width=25mm,minimum height=6mm,align=center}]
\node[box] (input) {Company Description};
\node[box,below=of input] (extract) {ExtractInfo};
\node[field,below left=of extract] (revenue) {Revenue};
\node[field,below=of extract] (employees) {Employees};
\node[field,below right=of extract] (industry) {Industry};
\node[box,below=15mm of employees] (category) {BusinessCategory};
\node[box,below=of category] (final) {Final Classification};
\draw[->,accent,thick] (input) -- (extract);
\draw[->,accent,thick] (extract) -- (revenue);
\draw[->,accent,thick] (extract) -- (employees);
\draw[->,accent,thick] (extract) -- (industry);
\draw[->,accent,thick] (revenue.south) |- (category.west);
\draw[->,accent,thick] (employees) -- (category);
\draw[->,accent,thick] (industry.south) |- (category.east);
\draw[->,accent,thick] (category) -- (final);
\end{tikzpicture}
\end{center}
\textbf{DSPy Predictor input:} Company Revenue \$3 Million; 120 employees; Retail.
\begin{lstlisting}[style=output]
@@PREDICTOR_OUTPUT@@
\end{lstlisting}
\textbf{Two-stage DSPy input:} ABC Retail Ltd.; annual revenue \$4 Million; 180 employees; Retail.
\begin{lstlisting}[style=output]
@@PIPELINE_OUTPUT@@
\end{lstlisting}

\section{Observations}
\begin{itemize}
\item CoT-style prompting produced the classification \texttt{@@COT_CLASS@@} with visible factors and a short rationale.
\item ToT-style prompting selected \texttt{@@TOT_CLASS@@} after listing alternatives rather than exposing private reasoning.
\item ReAct-style prompting separated the observable evaluation stages and returned \texttt{@@REACT_CLASS@@}.
\item DSPy Predictor returned \texttt{@@PREDICTOR_CLASS@@}; the pipeline returned \texttt{@@PIPELINE_CLASS@@} after independently extracting three fields.
\item The results show that declarative Signatures make the two-stage handoff explicit while keeping the same local Ollama runtime.
\end{itemize}
\newpage

\section{Result}
The practical successfully demonstrated structured CoT-style, ToT-style, and ReAct-style prompts together with a DSPy Predictor and a two-stage DSPy pipeline. All recorded outputs were generated by Qwen3 4B through Ollama on a GitHub-hosted Linux runner; no external inference API was used.
\end{document}
"""
    for marker, value in replacements.items():
        template = template.replace(marker, value)
    TEX_FILE.write_text(template, encoding="utf-8")


def print_consolidated_results(results):
    print("=== Consolidated Results ===")
    print("Technique | Classification | LLM Calls | Execution Time (s)")
    print("-" * 78)
    for technique, item in results.items():
        if technique != "dspy_lm":
            print(
                f"{technique} | {item['classification']} | {item['calls']} | {item['time']:.3f}"
            )
    print(f"\nCSV saved to: {RESULTS_FILE}")
    print(f"LaTeX report generated from actual outputs: {TEX_FILE}")


def main():
    print(f"{STUDENT_ID} - Generative AI Practical 3")
    print("Declarative Prompt Programming using CoT, ToT, ReAct and DSPy\n")
    print_environment()
    client = ollama.Client(host=OLLAMA_HOST)
    part_a = run_part_a(client)
    part_b = run_part_b()
    results = {**part_a, **part_b}
    write_results(results)
    build_latex(results)
    print_consolidated_results(results)


if __name__ == "__main__":
    main()
