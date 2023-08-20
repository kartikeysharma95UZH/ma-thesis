# Text Generation with Domain Adapted Models

This project demonstrates text generation using Hugging Face models for different language tasks. It includes three models: noun completion in English, noun completion in German, and phrase expansion in German.

## Setup

Clone the repository:
   ```
   git clone git@github.com:kartikeysharma95UZH/ma-thesis.git
   cd ma-thesis
   ```

Create a virtual environment and install dependencies from the requirements.txt file:
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
## Usage

1. `Noun Completion English`
This task involves completing an elided noun text by adding an appropriate text in English.

To run this model, use the following command:

```
python main.py --task noun-completion-en --input "Your input text here"
```

Replace "Your input text here" with the input text you want to generate results for.

2. `Noun Completion German`
This task involves completing an elided noun text(truncated nouns) by adding an appropriate text in German.

To run this model, use the following command:

```
python main.py --task noun-completion-de --input "Your input text here"
```

Replace "Your input text here" with the input text you want to generate results for.

3. `Phrase Expansion German`

This task involves expanding condensed coordinated soft-skill requirements using a phrase expansion model.

To run this model, use the following command:

```
python main.py --task phrase-expansion-de --input "Ihre Eingabetext hier"
```

Replace "Ihre Eingabetext hier" with the input text you want to generate results for.

## Example

Here's an example usage of the script for each of the task:


```
python main.py --task noun-completion-en --input "Heating and air conditioning technology"
```

```
python main.py --task noun-completion-de --input "Mittwoch- und Samstagnachmittag"
```

```
python main.py --task phrase-expansion-de --input "<SoftSkill_C>Kommunikations- </SoftSkill_C> und <SoftSkill>Teamfähigkeit</SoftSkill>"
```
