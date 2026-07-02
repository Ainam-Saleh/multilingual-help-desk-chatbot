import json
from simpletransformers.question_answering import QuestionAnsweringModel
import torch


def test_model():
    model_path = "outputs/nile_qa_model"
    print(f"Loading model from {model_path}...")

    use_cuda = torch.cuda.is_available()
    model = QuestionAnsweringModel("distilbert", model_path, use_cuda=use_cuda)

    # Test with different questions
    context_tuition = "Please visit https://www.nileuniversity.edu.ng/tuition-fees."
    question_tuition = "How much is the tuition fee?"

    context_accommodation = "No, the tuition fee is not accommodation inclusive. Visit https://nileuniversity.edu.ng/student-accommodation to see the tuition fee."
    question_accommodation = "Is accommodation included?"

    to_predict = [
        {
            "context": context_tuition,
            "qas": [
                {"question": question_tuition, "id": "0"}
            ]
        },
        {
            "context": context_accommodation,
            "qas": [
                {"question": question_accommodation, "id": "1"}
            ]
        }
    ]

    print("Making predictions...")
    predictions = model.predict(to_predict, n_best_size=1)

    # Unpack the tuple: (answers, probabilities)
    answers, probabilities = predictions

    print("\n" + "=" * 50)
    print("MODEL PREDICTIONS")
    print("=" * 50)

    for i, item in enumerate(to_predict):
        question_text = item['qas'][0]['question']

        # Get the answer - it's a dict with 'id' and 'answer' keys
        answer_dict = answers[i]
        # The 'answer' value is a list, so get the first element
        predicted_text = answer_dict['answer'][0]

        print(f"\nQ: {question_text}")
        print(f"A: {predicted_text}")
        print("-" * 50)


if __name__ == '__main__':
    test_model()
