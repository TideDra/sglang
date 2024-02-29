import argparse
import json
import os
import time
import uuid

import sglang as sgl
from sglang.test.test_utils import add_common_sglang_args_and_parse, select_sglang_backend

def load_questions(filename):
    questions = []
    with open(filename, "r") as fin:
        for line in fin:
            obj = json.loads(line)
            questions.append(obj)
    return questions

@sgl.function
def answer_if_eval(s,question):
    s += sgl.user(question)
    s += sgl.assistant(sgl.gen("answer"))

def write_answers(filename, model_id, questions, answers):
    with open(os.path.expanduser(filename), "w") as fout:
        for i in range(len(answers)):
            ans_json = {
                "prompt":questions[i]['prompt'],
                "response":answers[i]
            }
            fout.write(json.dumps(ans_json) + "\n")

def main(args):
    # Construct prompts
    questions = load_questions(args.question_file)
    arguments = [
        {"question": q['prompt']}
        for q in questions
    ]

    # Select backend
    backend = select_sglang_backend(args)
    sgl.set_default_backend(backend)

    # Run requests
    tic = time.time()
    rets = answer_if_eval.run_batch(
        arguments,
        temperature=0,
        max_new_tokens=256,
        num_threads=args.parallel,
        progress_bar=True)
    answers = [s['answer'] for s in rets]
    latency = time.time() - tic

    print(f"#questions: {len(questions)}, Latency: {latency:.2f}")

    # Write results
    model_id = backend.model_info["model_path"]
    answer_file = args.answer_file or f"data/input_response_data_{args.backend}.jsonl"
    write_answers(answer_file, model_id, questions, answers)
    args.num_questions = len(questions)
    with open(args.result_file, "a") as fout:
        value = {
            "task": "if_eval",
            "backend": args.backend,
            "num_gpus": 1,
            "latency": round(latency, 3),
            "num_requests": args.num_questions,
            "other": {
                "num_questions": args.num_questions,
                "parallel": args.parallel,
            }
        }
        fout.write(json.dumps(value) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-file", type=str, default="data/input_data.jsonl")
    parser.add_argument("--answer-file", type=str, default=None)
    args = add_common_sglang_args_and_parse(parser)
    main(args)
