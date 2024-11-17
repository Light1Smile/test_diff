from transformers import pipeline
import save_var

model_1="trick/false"
model_2="trick/false_2"

qa = pipeline(
    "question-answering",
    #model="NeuML/bert-small-cord19qa",
    model=model_1,
    tokenizer="NeuML/bert-small-cord19qa"
)

model_1="NeuML/bert-small-cord19-squad2"

qa = pipeline(
    "question-answering",
    #model="NeuML/bert-small-cord19qa",
    model=model_1,
    tokenizer="NeuML/bert-small-cord19qa"
)

qa = pipeline(
    "question-answering",
    #model="NeuML/bert-small-cord19qa",
    model=save_var.model_name,
    tokenizer="NeuML/bert-small-cord19qa"
)

model_1=model_2

qa = pipeline(
    "question-answering",
    #model="NeuML/bert-small-cord19qa",
    model=model_1,
    tokenizer="NeuML/bert-small-cord19qa"
)


def answer (query_text,context_text):
  answer = qa({
                "question": query_text,
                "context": context_text
               })
  print(answer)
  return answer