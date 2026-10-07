# TODOs

The current repository contains at the same time the code used to generate the experiments and results contained in the current Paper folder (a LaTex repository with ACL style for a short submission we are preparing) and what I have currently written up in the Paper folder. There are few things that are left at this point to complete the paper as well as to make the code "presentable".

## Missing Parts in Paper
- We need another section delving into BERT router using Integrated Gradient to actually see what elements make a router choose between Python or Lean: this is an important element which will complement the paper in a nice way, since this paper is a pilot study of a hybrid verification system based on dynamically chosen deterministic provers. My collaborator (who created this repo) told me that the visualisations containing this should already exist in this folder, look for them but if you do not find or if they are not convincing then recompute the integrated gradient visualisation yourself to show more clearly the elements on which the router focuses.

- The current table describing results put everything together but results are actually not comparable: the BERT results are computed on test set, while the LLM results on the entire ProcessBench. We need to create another table where we put the BERT results and add the LLM router results but computed only on test set this time: to do so we need to identify the LLM-router results in the current repo, identify the test split and then just re-compute the metrics with existing results on that split.

- Following from above, we need to add the correct description of the experiments in experimental setup (the results and conclusion section I have not written yet, but I will do it myself). Every time you write something in the tex files, always do not be disruptive and follow my style as much as possible while not being too long in your added parts.

- All of the appendices are currently missing: please include the required appendices documenting technical details like the parameters used to fit BERT and the model cards in each case, data statistics, etc.

- Assess at this point the quality of the paper: what baselines might be added to make it better? Does everything look sound? How would you score this paper in an ACL conference? (remember this is a short paper)

## Re-arranging
- You need to refactor the code in order for it to be presentable as the accompanying code of the paper.

- In the Paper folder, we have a lot of redundant and useless .tex files because I copied this project from a previous project of mine: remove all the .tex files and figures and tables which do not belong to this project.
