import json
import logging
import torch
from simpletransformers.question_answering import QuestionAnsweringModel, QuestionAnsweringArgs


# ============================================================
# MODEL TRAINING SCRIPT
# This script takes the FAQ dataset and uses it to fine-tune
# a DistilBERT model specifically for Nile University questions.
#
# WHY WE TRAIN:
# DistilBERT is pre-trained on general English text (Wikipedia,
# books, etc). It doesn't know anything about Nile University.
# Fine-tuning teaches it to understand university-specific
# questions and extract answers from our FAQ dataset.
#
# HOW TO RUN:
# python model_train.py
# Training may take several minutes depending on your hardware.
# The trained model is saved to outputs/nile_qa_model/
# ============================================================


def train_qa_model():

    # --------------------------------------------------------
    # STEP 1: LOAD THE DATASET
    # Reads the comprehensive FAQ JSON file which contains
    # 351 question-answer pairs about Nile University.
    # This is the data the model will be trained on.
    # --------------------------------------------------------
    input_file = 'nile_dataset_comprehensive.json'
    print(f"Loading dataset from {input_file}...")

    with open(input_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # --------------------------------------------------------
    # STEP 2: TRANSFORM DATA TO SIMPLETRANSFORMERS FORMAT
    # The simpletransformers library expects data in a specific
    # SQuAD-style format (the same format used by Stanford's
    # Question Answering Dataset).
    #
    # Each entry must contain:
    # - 'context'  → the passage the model reads to find the answer
    # - 'qas'      → list of questions about that context, each with:
    #     - 'id'          → unique identifier for the question
    #     - 'question'    → the question text
    #     - 'answers'     → list of correct answers, each with:
    #         - 'text'         → the answer text
    #         - 'answer_start' → character position where answer starts
    #     - 'is_impossible' → False means the answer exists in context
    #
    # In our case, since each FAQ entry IS its own context,
    # the answer is always the full context starting at position 0.
    # --------------------------------------------------------
    train_data = []
    for idx, item in enumerate(raw_data):
        context_text = item['answer']    # The answer becomes the context passage
        question_text = item['question'] # The question the model needs to learn

        entry = {
            'context': context_text,
            'qas': [
                {
                    'id': str(idx),          # Unique ID for each Q&A pair
                    'question': question_text,
                    'answers': [
                        {
                            'text': context_text,  # Full answer is the correct span
                            'answer_start': 0       # Answer starts at position 0
                        }
                    ],
                    'is_impossible': False  # All our questions have answers
                }
            ]
        }
        train_data.append(entry)

    # --------------------------------------------------------
    # STEP 3: CONFIGURE TRAINING ARGUMENTS
    # These settings control how the training process runs.
    #
    # train_batch_size    → how many examples processed at once.
    #                       Higher = faster but needs more RAM.
    #                       32 is a good balance for most machines.
    #
    # num_train_epochs    → how many times the model sees the full
    #                       dataset during training. 30 epochs gives
    #                       the model enough time to learn the FAQ
    #                       content thoroughly.
    #
    # evaluate_during_training → disabled to speed up training since
    #                            we don't have a separate test set.
    #
    # overwrite_output_dir → if a previous model exists in the output
    #                        folder, overwrite it with the new one.
    #
    # save_model_every_epoch → disabled to avoid filling up disk space
    #                          with intermediate model checkpoints.
    #
    # output_dir → folder where the final trained model will be saved.
    # --------------------------------------------------------
    model_args = QuestionAnsweringArgs()
    model_args.train_batch_size = 32
    model_args.num_train_epochs = 30
    model_args.evaluate_during_training = False
    model_args.overwrite_output_dir = True
    model_args.save_model_every_epoch = False
    model_args.output_dir = "outputs/nile_qa_model"

    # NOTE FOR WINDOWS USERS:
    # If you get multiprocessing errors during training on Windows,
    # uncomment the line below to force single-process data loading.
    # model_args.process_count = 1

    # --------------------------------------------------------
    # STEP 4: INITIALIZE THE MODEL
    # Loads the base DistilBERT model from HuggingFace.
    # "distilbert-base-uncased" is the pre-trained starting point
    # before our fine-tuning begins.
    #
    # use_cuda → automatically uses GPU if one is available on
    #            your machine, otherwise falls back to CPU.
    #            GPU training is significantly faster.
    # --------------------------------------------------------
    use_cuda = torch.cuda.is_available()
    model = QuestionAnsweringModel(
        "distilbert",
        "distilbert-base-uncased",  # Base pre-trained model from HuggingFace
        args=model_args,
        use_cuda=use_cuda
    )

    # --------------------------------------------------------
    # STEP 5: TRAIN THE MODEL
    # This is where the actual learning happens.
    # The model repeatedly reads all 351 FAQ pairs across 30 epochs,
    # adjusting its internal weights each time to get better at
    # extracting the correct answer span from the context.
    # The final trained model is saved to outputs/nile_qa_model/
    # --------------------------------------------------------
    print("Starting training...")
    model.train_model(train_data)
    print(f"Training complete. Model saved to {model_args.output_dir}")


# ============================================================
# CRITICAL FIX FOR WINDOWS — if __name__ == '__main__'
# On Windows, Python's multiprocessing module requires that
# the training code runs inside this block.
# Without it, Windows creates infinite child processes and
# crashes with a RuntimeError.
# This block ensures training only starts from the main process,
# not from any worker processes spawned during training.
# This is NOT needed on Linux or Mac but doesn't cause harm there.
# ============================================================
if __name__ == '__main__':
    train_qa_model()