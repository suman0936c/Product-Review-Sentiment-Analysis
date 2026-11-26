# Viva Questions and Answers

1. **What is sentiment analysis?** It identifies opinion in text. Here it predicts positive or negative reviews.
2. **What is NLP?** Natural Language Processing helps computers work with human language.
3. **Why clean text?** It removes inconsistent case, links, punctuation, and extra spaces.
4. **Why keep words such as not?** Negation can change the meaning of a review.
5. **Why remove duplicates?** Duplicate reviews can bias training and evaluation.
6. **Why exclude rating 3?** It is usually neutral or mixed, while this is a binary task.
7. **What is TF-IDF?** A numerical representation that gives higher importance to useful, less-common words.
8. **Why vectorize text?** ML models need numbers rather than raw text.
9. **What is data leakage?** Training using information from test data. TF-IDF is fitted only on training text.
10. **What is a train test split?** Training data teaches the model and test data evaluates unseen reviews.
11. **What is Logistic Regression?** A classifier that estimates a class probability.
12. **What is ANN?** An Artificial Neural Network learns patterns through connected layers.
13. **What is a neuron?** A small unit that combines inputs and applies an activation function.
14. **What is ReLU?** An activation that returns zero for negative input and keeps positive input.
15. **What is Sigmoid?** It converts output to a value between 0 and 1.
16. **What is Dropout?** It temporarily ignores some neurons during training to reduce overfitting.
17. **What is an epoch?** One full pass through training data.
18. **What is batch size?** The number of examples used before updating weights.
19. **What is Adam?** An optimizer that updates model weights to reduce loss.
20. **What is binary crossentropy?** A loss function for two-class probability prediction.
21. **What is overfitting?** Learning training data too closely and performing poorly on new data.
22. **What is accuracy?** The proportion of correct predictions.
23. **What is precision?** The proportion of predicted positives that are correct.
24. **What is recall?** The proportion of actual positives correctly found.
25. **What is F1-score?** A balanced measure of precision and recall.
26. **What is a confusion matrix?** A table of actual versus predicted classes.
27. **Why compare models?** It shows whether the ANN improves on a simple baseline.
28. **How does the app predict a review?** It cleans text, applies saved TF-IDF, then sends features to the saved ANN.
