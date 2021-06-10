To run the statistical language model file, type: python StatisticalLanguageModel.py [smoothing type] [corpus_path]
where smoothing type = k for kneser ney
and   smoothing type = w for witten bell

It takes an input sentence , and returns the probability score of the sentence based on the corpus provided.


To run the Neural Language model,open the file "Pytorch_Neural_Language_Model.ipynb" in a jupyter notebook, train the model, and then load the saved model after training, put the sentence for which you want to find the perplexity in a list, and pass it to the "evaluateNLM()" function present at the end, as its third parameter, it will return the loss, perplexity and accuracy in a list.
It will require torchtext 0.6.0 to be installed and other ML libraries like torch,nltk and sklearn.