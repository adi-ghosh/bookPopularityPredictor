# Book Popularity Prediction

## CSDS 438 Group 3
Adi Ghosh, Xiao Wang, Sarah Kurelowech, and Sarah Kugelmas
[@SarahKugelmas](https://github.com/SarahKugelmas)

### Contents
1. [Problem Statement](#problem-statement)
2. [Relevant Background](#relevant-background)
3. [Dataset Overview](#dataset-overview)
4. [Methodology](#methodology)
5. [Findings](#findings)
6. [Conclusions](#conclusions)
7. [Future Work](#future-work)

### Problem Statement
Can we predict the popularity of a book before it is published based on previous successful works using an AI model?

### Relevant Background
- **Bag of Words (BoW)**: Simplifies a book's representation by counting word frequencies, disregarding grammar and order.
- **HPC (High-Performance Computing)**: Necessary for running BoW on entire books and training AI models quickly, especially for parallel threading and GPU utilization.

### Dataset Overview
- Consists of 24 American classic books.
- Each book entry includes title, author, year of publication, average rating from multiple websites, and full text.
- Example dataset segment provided.

### Methodology
- **Data Cleaning**: Ensure text is in one line per book.
- **Models Implemented**: Bidirectional RNN, SVM, Logistic Regression, GaussianNB, MLP Classifier.
- **Model Tuning**: Adjust parameters for each model.
- **HPC Environment**: Utilized Case Western's Markov HPC Cluster for efficient processing.
- **Findings**: Inconclusive results with accuracy scores of 0.0 across all models.

### Findings
- Results inconclusive; all models yielded 0.0 accuracy.
- Likely causes include inadequate data and lack of direct correlation between words and book popularity.
- HPC run information provided.

### Conclusions
- Predicting book popularity based solely on textual content appears unfeasible.
- HPC enabled quick processing of large text datasets and complex ML models, highlighting its importance.

### Future Work
- Experiment with different models and parameters to improve accuracy.
- Explore scalability with varying numbers of worker threads.
- Increase dataset size and consider extracting themes from text.
- Develop better quantitative measures of popularity.
