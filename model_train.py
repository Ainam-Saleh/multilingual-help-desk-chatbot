import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)

# ============================================================
# FLAN-T5 MODEL TRAINING SCRIPT
# This is an alternative training script that uses Google's
# FLAN-T5 model instead of DistilBERT.
#
# DIFFERENCE FROM model_train.py:
# - model_train.py uses DistilBERT → EXTRACTIVE approach
#   (finds and returns a span of text from the context)
# - This script uses FLAN-T5 → GENERATIVE approach
#   (generates a brand new answer sentence from scratch)
#
# WHY FLAN-T5:
# FLAN-T5 is an instruction-tuned model, meaning it was
# pre-trained to follow instructions like "answer this question".
# This makes it better at understanding varied phrasings and
# producing natural-sounding answers compared to DistilBERT.
#
# HOW TO RUN:
# python model2.py
# The trained model is saved to ./nile_university_qa_model/
# ============================================================


# --------------------------------------------------------
# CONFIGURATION
# Central place to change key settings without digging
# through the code:
# MODEL_ID  → the pre-trained model to start from (HuggingFace)
# DATA_FILE → the FAQ dataset to train on
# OUTPUT_DIR → where to save the trained model when done
# --------------------------------------------------------
MODEL_ID = "google/flan-t5-base"  # Strong instruction-tuned base model
DATA_FILE = "nile_dataset.json"
OUTPUT_DIR = "./nile_university_qa_model"


def train():

    # --------------------------------------------------------
    # STEP 1: LOAD AND SPLIT THE DATASET
    # Reads the FAQ JSON file and converts it into a HuggingFace
    # Dataset object, which is the standard format expected by
    # the Transformers training library.
    #
    # We split the data 90/10:
    # - 90% used for training (model learns from these)
    # - 10% used for evaluation (model is tested on these
    #   at the end of each epoch to track how well it's doing)
    # --------------------------------------------------------
    with open(DATA_FILE, 'r') as f:
        data = json.load(f)

    # Convert list of Q&A dicts into HuggingFace Dataset format
    dataset = Dataset.from_list(data)

    # Split into train and test sets
    dataset = dataset.train_test_split(test_size=0.1)

    # --------------------------------------------------------
    # STEP 2: TOKENIZATION AND PREPROCESSING
    # Loads the tokenizer that matches our chosen model.
    # The tokenizer converts raw text into numbers (token IDs)
    # that the model can process.
    #
    # preprocess_function does two things:
    # (a) Formats each question with the task prefix
    #     "answer this question: ..." — this is how FLAN-T5
    #     expects instructions to be phrased
    # (b) Tokenizes both the input questions and target answers
    #     with a maximum length to keep memory usage manageable
    #
    # max_length=128 for questions → questions are usually short
    # max_length=256 for answers   → answers may be longer
    # padding="max_length"         → pads shorter texts to the
    #                                same length so they can be
    #                                batched together efficiently
    # --------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    def preprocess_function(examples):
        # Add task instruction prefix that FLAN-T5 understands
        inputs = [f"answer this question: {q}" for q in examples["question"]]

        # Tokenize the input questions
        model_inputs = tokenizer(
            inputs,
            max_length=128,
            truncation=True,
            padding="max_length"
        )

        # Tokenize the target answers separately as labels
        # text_target tells the tokenizer these are output sequences
        labels = tokenizer(
            text_target=examples["answer"],
            max_length=256,
            truncation=True,
            padding="max_length"
        )

        # Attach tokenized answers as labels for the model to learn from
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Apply preprocessing to the entire dataset in batches
    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    # --------------------------------------------------------
    # STEP 3: MODEL AND DATA COLLATOR SETUP
    # Loads the FLAN-T5 base model with its pre-trained weights.
    # AutoModelForSeq2SeqLM is used because FLAN-T5 is a
    # sequence-to-sequence model — it takes a sequence (question)
    # and generates another sequence (answer).
    #
    # DataCollatorForSeq2Seq handles batching during training:
    # it dynamically pads sequences within each batch to the
    # same length, which is more memory efficient than fixed padding
    # --------------------------------------------------------
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # --------------------------------------------------------
    # STEP 4: TRAINING ARGUMENTS
    # These settings control how the training process behaves.
    #
    # eval_strategy="epoch"     → evaluate model after every epoch
    # save_strategy="epoch"     → save a checkpoint after every epoch
    # learning_rate=3e-4        → slightly higher than default because
    #                             our dataset is small; helps the model
    #                             learn faster without overshooting
    # per_device_train_batch_size=8 → 8 examples processed per step.
    #                                 Lower than DistilBERT (32) because
    #                                 FLAN-T5 is a larger model and needs
    #                                 more memory per example
    # num_train_epochs=15       → 15 passes through the dataset.
    #                             Good balance for ~100-350 examples
    # weight_decay=0.01         → regularization to prevent overfitting
    #                             (stops the model memorizing answers
    #                             instead of learning to generalize)
    # save_total_limit=2        → only keep the 2 most recent checkpoints
    #                             to avoid filling up disk space
    # predict_with_generate=True → use the model's generation capability
    #                              during evaluation (important for T5)
    # load_best_model_at_end=True → after all epochs, automatically load
    #                               the checkpoint with the best eval score
    # --------------------------------------------------------
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=15,
        weight_decay=0.01,
        save_total_limit=2,
        predict_with_generate=True,
        logging_steps=10,
        load_best_model_at_end=True,
    )

    # --------------------------------------------------------
    # STEP 5: INITIALIZE THE TRAINER
    # The Seq2SeqTrainer handles the entire training loop:
    # - Forward pass (model makes predictions)
    # - Loss calculation (how wrong the predictions are)
    # - Backward pass (adjusting model weights to improve)
    # - Evaluation at the end of each epoch
    # - Saving checkpoints
    #
    # We pass in:
    # - model          → the FLAN-T5 model to train
    # - args           → training configuration from Step 4
    # - train_dataset  → the 90% training split
    # - eval_dataset   → the 10% evaluation split
    # - tokenizer      → needed for saving and generation
    # - data_collator  → handles dynamic padding during batching
    # --------------------------------------------------------
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # --------------------------------------------------------
    # STEP 6: TRAIN AND SAVE THE MODEL
    # trainer.train() runs the full training loop across all epochs.
    # Once complete, the best model checkpoint is saved along with
    # the tokenizer so both can be loaded together for inference.
    # The saved model can then be loaded in the chatbot using:
    # AutoModelForSeq2SeqLM.from_pretrained("./nile_university_qa_model")
    # --------------------------------------------------------
    print("Starting training...")
    trainer.train()

    print(f"Saving final model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)      # Save model weights and config
    tokenizer.save_pretrained(OUTPUT_DIR)  # Save tokenizer alongside model


# ============================================================
# SCRIPT ENTRY POINT
# Ensures training only starts when this file is run directly.
# Prevents accidental execution if this file is imported
# as a module by another script.
# ============================================================
if __name__ == "__main__":
    train()