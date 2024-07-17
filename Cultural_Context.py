from transformers import BertTokenizer, BertForTokenClassification, BertForQuestionAnswering
from ampligraph.latent_features import TransESimple
from ampligraph.evaluation import evaluate_link_prediction
from gensim.models import KeyedVectors
import torch
import numpy as np

# Load pre-trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-large-cased-finetuned-conll03-english')
model = BertForTokenClassification.from_pretrained('dbmdz/bert-large-cased-finetuned-conll03-english')
qa_model = BertForQuestionAnswering.from_pretrained('distilbert-base-cased-distilled-squad')

# Load pre-trained knowledge graph embeddings
kge_model = TransESimple(batches_count=100, epochs=500, k=200, eta=50,
                         optimizer='adagrad', optimizer_params={'lr': 0.1},
                         loss='multiclass_nll', regularizer='none',
                         regularizer_params={'lambda': 1e-5},
                         negative_sample_size=1, verbose=True, random_seed=42)

def detect_entities(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    predictions = outputs[0][0].argmax(-1).tolist()
    entities = []
    for i, prediction in enumerate(predictions):
        if prediction != 0:  # 0 corresponds to 'O'
            entity = tokenizer.convert_ids_to_tokens(i)
            entities.append((entity, prediction))
    return entities

def get_kge_info(entity):
    try:
        entity_index = kge_model.entity_index[entity]
        return kge_model.entity_vectors[entity_index]
    except KeyError:
        return None

def get_context(context, question):
    inputs = qa_model.prepare_question_answer(question, context)
    outputs = qa_model(**inputs)
    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits
    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores)
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end+1]))
    return answer

def calculate_similarity(entity1, entity2):
    vec1 = get_kge_info(entity1)
    vec2 = get_kge_info(entity2)
    if vec1 is not None and vec2 is not None:
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    else:
        return None
