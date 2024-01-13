Here is RoBERTa transformer from Hugging Face, constructed for document classification model. See dcm_roberta_clean.ipynb for more details.

Some information:
1. it is RoBERTa 'roberta-base' transformer from Hugging Face (https://huggingface.co/roberta-base)
2. it is constructed on the local dataset, 75% of data for training, rest of 25% data for testing.
3. accuracy 0.96 (96%)
4. roberta_dcm.pkl is the binary serialization of the model (after training and testing)
5. the predicted output is a numerical value - see the substitution dictionary to obtain the predicted document class - string value
6. The training and testing was done on unpreprocessed data (so raw data from files); predictions should be done in the same way

