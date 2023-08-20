# Master's Thesis: Using Large Language Models (LLMs) to Expand Condensed Coordinated German and English Expressions into Explicit Paraphrases

Welcome to the GitHub repository for my Master's thesis from the Department of Computational Linguistics at the University of Zurich. This repository contains the code, data, and documents associated with my thesis.

## Abstract

This master’s thesis explores fine-tuning Large Language Models (LLMs) to reformulate condensed coordinated expressions found in job postings. This kind of condensed coordinated expression is frequently used in job postings, which is our target text genre for this work. Four gold-standard datasets were created for two tasks in English and German.

The first task focuses on truncated word completion, where elided text like “Haus- und Gartenarbeit” (house and garden work) needs to be completed to “Hausarbeit und Gartenarbeit.” The German GS dataset consists of 510 samples, while the English GS contains 402 samples. The primary goal is to assess the LLMs’ performance in this task and identify promising models for the second, more complex task.

The second task involves expanding condensed coordinated soft-skill requirements
like “Sie arbeiten sehr selbständig, ziel- und kundenorientiert” into explicit self-contained paraphrases such as “Sie arbeiten sehr selbständig, arbeiten zielorientiert und arbeiten kundenorientiert”. To achieve a proper mapping of soft-skill requirements to a detailed domain ontology, it is crucial to provide self-contained text spans that refer to a single concept. For creating the German GS, we utilized In-Context Learning with ChatGPT, providing 5 examples in the prompt to generate additional samples. Subsequently, these samples were used to fine-tune GPT-3 and later manually verified to form a GS dataset comprising 1968 samples.

In the first task, T5-large, and FLAN-T5-large, and GPT models showed similar levels of accuracy. However, in the second task, T5-large and FLAN-T5-large performed poorly. To improve results, we applied PEFT-based techniques, LORA, to fine-tune BLOOM, T5-Large, FLAN T5-XXL, and mT5-XL on a single GPU. Among these, GPT-3 demonstrated superior performance, closely followed by mT5-XL in overall evaluations. For evaluation, we measured how incomplete soft skill text spans were completed, assessed both completed and incomplete soft skills, and evaluated overall sentence similarity. Error metrics such as Rouge-L, average Levenshtein distance, % of matched skills, and Cosine Similarity were used to evaluate soft skill changes and overall text similarity. In conclusion, Large Language Models (LLMs) effectively expanded condensed coordinated expressions into simpler formulations, including completing hyphenated words in German, without relying on traditional methods sensitive to grammatical and spelling errors.

## Zusammenfassung

Diese Arbeit untersucht die Fähigkeit von Large Language Models (LLMs), kondensierte koordinierte Ausdrücke in Stellenanzeigen expliziter zu reformulieren. Diese Art Ausdrücke wird häufig in Stellenausschreibungen verwendet, die zugrundeliegende Textgattung für diese Arbeit sind. Vier Gold-Standard-Datensätze wurden für je zwei Aufgaben in Englisch und Deutsch erstellt.

Die erste Aufgabe konzentriert sich auf die Vervollständigung von verkürzten Wörtern, zum Beispiel soll “Haus- und Gartenarbeit” zu “Hausarbeit und Gartenarbeit” ergänzt werden. Der deutsche Datensatz besteht aus 510 unterschiedlichen Beispielen, der englische aus 402. Damit soll die Leistung der LLMs in dieser Aufgabe erhoben und geeignete Modelle für die zweite, komplexere Aufgabe identifiziert werden.

In der zweiten Aufgabe werden kondensierte koordinierte Soft-Skill-Anforderungen wie Sie arbeiten sehr selbständig, ziel- und kundenorientiert in explizite, in sich geschlossene Paraphrasen wie Sie arbeiten sehr selbständig, arbeiten zielorientiert und arbeiten kundenorientiert erweitert. Um eine korrekte Abbildung von Soft-Skill-Anforderungen auf eine detaillierte Domänenontologie zu erreichen, ist es entscheidend, inhaltlich Textabschnitte bereitzustellen, die sich auf ein einzelnes Konzept beziehen. Für die Erstellung des deutschen GS haben wir In-Context Learning mit ChatGPT verwendet, mit jeweils 5 Beispielen in der Eingabe. Danach wurde die manuell korrigierte Ausgabe iterativ für das Optimieren von GPT-3-basierten Modellen verwendet und letztlich ein Datensatz mit 1’968 Beispielen erstellt.

Die erste Aufgabe lösten die Modelle T5-large und FLAN-T5-large sowie GPT mit hoher Genauigkeit. Bei der zweiten Aufgabe jedoch schnitten T5-large und FLAN-T5-large schlecht ab. Bessere Resultate erhielten wir mit PEFT-basierten Feinabstimmungstechniken von BLOOM, T5-Large, FLAN T5-XXL und mT5-XL an. GPT-3 zeigte die beste Leistung, dicht gefolgt von mT5-XL. Für die Bewertung haben wir folgendes gemessen: die Ergänzung von unvollständigen Soft-Skill-Segmenten, die Ähnlichkeit aller (auch vollständigen) Segmente, sowie die Ähnlichkeit des ganzen Satzes. Metriken wie Rouge-L, die Levenshtein-Distanz, % der übereinstimmenden Fertigkeiten und die Cosinus-Ähnlichkeit wurden zur Bewertung der Soft-Skill-Änderungen und der Gesamttextähnlichkeit verwendet. Zusammenfassend lässt sich sagen, dass LLMs kondensierte koordinierte Ausdrücke effektiv in einfachere Formulierungen umwandeln können, einschliesslich der Vervollständigung von Wörtern mit Auslassungsstrichen im Deutschen, ohne auf herkömmliche morphologische und korpusstatistische Methoden zurückgreifen zu müssen.

## Repository Contents

- `Data-Sampler/`: This directory contains the code implementation of the Nilsimsa sampler to select unique cases from the extracted data for dataset creation.
- `Fine-Tune-pipeline/`: This directory contains the code implementation of the entire project pipeline for fine-tuning several LLMs. 
- `Gold-Standard-data-prep/`: This directory contains the code implementation of the data extraction from the raw job ads for both the tasks of Noun Completion and Phrase Expansion 
- `Model-Evaluation-&-Analysis/`: Here, you'll find the Python notebooks containing the evaluation of several models on three different evaluation categories. For each evaluation category, several error metrics are used, and results have been analyzed for further seven problem types 
- `PEFT-LoRA/`: This directory contains the code implementation of fine-tuning various LLMs using a PEFT technique, LoRA (Low-Rank Adaptation of Large Language Models)
- `data_files/`: Here, you'll find the Gold Standard datasets in German and English and any other relevant data files.
- `MA_Thesis_Kartikey_Sharma`: This document contains my full thesis document.

## Installation and Setup

1. Clone this repository: `git clone git@github.com:kartikeysharma95UZH/ma-thesis.git`
2. For Model Inference, navigate to the `Model-inference` directory: `cd Model-inference`
3. Install the required dependencies: 

```
pip install -r requirements.txt
```
4. Run the project: `*******`

## Usage

To [`description of what can be achieved`], follow these steps:

1. [Step 1]
2. [Step 2]
3. [Step 3]

## Thesis Document

The full thesis document is available for download. You can access it here: [Download Thesis](MA_Thesis_Kartikey_Sharma.pdf).

## Contributing

If you're interested in contributing to this project, feel free to reach out to me.

## Contact Information

If you have any questions or feedback, you can reach out to me through my GitHub profile.

