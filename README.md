# Binding-Affinity-of-SARS-CoV-2
A neural network based off MHCSeqNet that predicts binding affinity between peptides and MHC-I alleles, and its application to the coronavirus.

The .ipynb files are not displaying their results, so I've attached their conversions to .py. To train the model:
1. Create the training data by running embed_sequences.ipynb to embed the peptide and allele sequences in our dataset. This uses MIT Professors Tristan Bepler and Bonnie Berger's model, "Learning protein sequence embeddings using information from structure." (Of course, this will work if you run the .py files rather than .ipynb.)
2. Run train_network.ipynb

To evaluate the trained model's predictions on the coronavirus:
1. Split the coronavirus amino acid sequence into peptides with get_coronavirus_peptides.ipynb
2. Embed the peptides with embed_coronavirus_peptides.ipynb
3. Evaluate with eval_on_coronavirus.ipynb

eval_on_coronavirus.pdf shows our results in PDF format.
