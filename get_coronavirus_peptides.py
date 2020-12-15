#!/usr/bin/env python
# coding: utf-8

# In[ ]:



from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


import pandas as pd
import os
os.chdir('/content/drive/My Drive/Peptide MHC Project/Coronavirus')


# In[ ]:


corona = pd.read_csv("coronavirus_netchop.csv")
aa_indices = corona.index[corona['pos'] == 1].tolist()

seq_lens = list(range(8, 16))
count = 0
ranges = []
pep_dict = {}


# In[ ]:


# Each peptide sequence length
for k in seq_lens:
    # thresh = 0.7 if k < 11 else 0.8
    thresh = .7
    # Each protein
    for j in range(len(aa_indices)):
        if j < len(aa_indices) - 1:
            end_index = aa_indices[j + 1]
        else:
            end_index = corona.shape[0]
        # Each amino acid
        for stop in range(aa_indices[j] + k - 1, end_index):
            if corona['score'].iloc[stop] < thresh:
                # Not at a splicing place
                continue
            if stop >= aa_indices[j] + k:
                # At a splicing place, but not for a k-mer
                if corona['score'].iloc[stop-k] < thresh:
                    continue

            # Start and stop amino acids in the k-mer are above splice threshold
            start = stop - k + 1
            peptide_range = [start, stop]
            # Insert peptide_range into lists
            if len(ranges) == 0:
                ranges.append(peptide_range)
            else:
                for i in range(len(ranges)):
                    if stop < ranges[i][0]:
                        ranges.insert(i, peptide_range)
                        break
                    if i == len(ranges) - 1:
                        ranges.append(peptide_range)

            kmer = "".join(corona['AA'].iloc[start:(stop + 1)].to_list())
            # Cannot store same peptide multiple times
            pep_dict[kmer] = peptide_range

print(len(ranges))  # total number of extracted peptides
print(len(pep_dict))  # number of unique peptides


# In[ ]:


# Take just the peptides that immediately precede and follow another plausible peptide
best_peptides = []
best_ranges = []
total = len(ranges)
for p in range(total):
    # False unless starts at beginning of protein's amino acid sequence
    first_aa = sum([ranges[p][0] == index for index in aa_indices])
    immediately_following = True if first_aa else False

    # False unless stops at end of protein's amino acid sequence
    last_aa = sum([ranges[p][1] == index - 1 for index in aa_indices])
    immediately_preceding = True if last_aa else False
    for i in range(10):
        if not immediately_following or not immediately_preceding:
            if p - i >= 0:
                if ranges[p - i][1] == ranges[p][0] - 1:
                    immediately_following = True
            if p + i < total:
                if ranges[p + i][0] == ranges[p][1] + 1:
                    immediately_preceding = True
    if immediately_following and immediately_preceding:
        for pep, pep_range in pep_dict.items():
            if pep_range == ranges[p]:
                best_peptides.append(pep)
                best_ranges.append(pep_range)


# In[ ]:


num_best = len(best_peptides)
print(num_best)
print(best_ranges[:20])
print(best_peptides[:20])

# if not os.path.exists("corona_peptides"):
#   os.mkdir("corona_peptides")
with open("all_corona_peptides.txt", "w") as f:
    for pep in best_peptides:
        f.write(pep + '\n')


# In[ ]:




