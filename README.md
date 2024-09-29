# Libraries to Install -
numpy, torch, transformers, colorama, joblib, scikit-learn

# Code to run -

## Command for extracting Intra and Inter Sentences Biases score on original models

cd code/

python3 eval_discriminative_models.py --pretrained-class bert-base-cased --tokenizer BertTokenizer --intrasentence-model BertLM --intersentence-model BertNextSentence --input-file ../data/dev.json --output-dir predictions/

## Command for extracting Intra and Inter Sentences Biases score while pruning models

cd code/

python3 eval_pruned_discriminative_models.py --pretrained-class bert-base-cased --tokenizer BertTokenizer --intrasentence-model BertLM --intersentence-model BertNextSentence --input-file ../data/dev.json --output-dir predictions/ --do_pruning

## Command for Computing LMS, SS, CAT scores 

cd code/

python3 evaluation.py --gold-file ../data/dev.json --predictions-dir predictions/

