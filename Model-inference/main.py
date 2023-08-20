import argparse
from text_generation_script import TextGenerationModel

def main(args):
    task_name = args.task_name
    input_text = args.input_text

    model = TextGenerationModel(task_name)
    output_text = model.generate_text(input_text)
    print(f"Input: {input_text}\nOutput: {output_text}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text Generation Script")
    parser.add_argument("--task", dest="task_name", required=True, help="Task name: noun-completion-en, noun-completion-de, or phrase-expansion-de")
    parser.add_argument("--input", dest="input_text", required=True, help="Input text for generation")
    
    args = parser.parse_args()
    main(args)
