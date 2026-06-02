"""
The goal of the analyser is to reason purely over the medical signals extracted by the
extractor agent in step 1. Either using the models to perform a one-shot reasoning
or to force a chain-of-thought reasoning over the extracted signals. In terms of choosing
the best method for this, leveraging a chain-of-thought approach is more reasonable since 
we are dealing with medical diagnoses and clinical signals
"""