from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import argparse


class TextGenerationModel:
    def __init__(self, task_name):
        self.task_name = task_name
        if task_name == "noun-completion-en":
            self.tokenizer = AutoTokenizer.from_pretrained(
                "Kartikey95/flan-t5-large-noun-completion-en"
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                "Kartikey95/flan-t5-large-noun-completion-en"
            )
        elif task_name == "noun-completion-de":
            self.tokenizer = AutoTokenizer.from_pretrained(
                "Kartikey95/flan-t5-large-noun-completion-de"
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                "Kartikey95/flan-t5-large-noun-completion-de"
            )
        elif task_name == "phrase-expansion-de":
            self.config = PeftConfig.from_pretrained(
                "Kartikey95/mt5-xl-phrase-expansion-de"
            )
            self.base_model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-xl")
            self.model = PeftModel.from_pretrained(
                self.base_model, "Kartikey95/mt5-xl-phrase-expansion-de"
            )
            self.tokenizer = AutoTokenizer.from_pretrained("google/mt5-xl")
        else:
            raise ValueError("Invalid task name")

    def generate_text(self, input_text):
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        output_ids = self.model.generate(input_ids=input_ids, max_new_tokens=280)
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return output_text


def main(args):
    task_name = args.task_name
    input_text = args.input_text

    model = TextGenerationModel(task_name)
    output_text = model.generate_text(input_text)
    print(f"Input: {input_text}\nOutput: {output_text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text Generation Script")
    parser.add_argument(
        "--task",
        dest="task_name",
        required=True,
        help="Task name: noun-completion-en, noun-completion-de, or phrase-expansion-de",
    )
    parser.add_argument(
        "--input", dest="input_text", required=True, help="Input text for generation"
    )

    args = parser.parse_args()
    main(args)
