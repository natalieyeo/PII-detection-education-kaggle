### Detecting Personally Identifiable Information From Student Writing

**Natalie Yeo**

#### Executive summary

In this project, we develop a model to label Personally Identifiable Information (PII) in student writing. This prompt is inspired by the Kaggle Competition hosted by the Learning Agency Lab. 

#### Rationale

In today’s era of abundant educational data from sources such as ed tech, online learning, and research, widespread PII is a key challenge. PII’s presence is a barrier to analyze and create open datasets that advance education because releasing the data publicly puts students at risk. To reduce these risks, it’s crucial to screen and cleanse educational data for PII before public release, which data science could streamline.

Manually reviewing the entire dataset for PII is currently the most reliable screening method, but this results in significant costs and restricts the scalability of educational datasets. While techniques for automatic PII detection that rely on named entity recognition (NER) exist, these work best for PII that share common formatting such as emails and phone numbers. PII detection systems struggle to correctly label names and distinguish between names that are sensitive (e.g., a student's name) and those that are not (e.g., a cited author).

Reliable automated techniques could allow researchers and industry to tap into the potential that large public educational datasets offer to support the development of effective tools and interventions for supporting teachers and students.

#### Research Question

What is the best model for detecting custom Personally Identifiable Information in educational datasets? How do deep learning models compare with transformers compare to Microsoft Presidio Analyzer, a named entity recognizer which combines pattern recognition and natural language processing? Our goal is to find the parameters of a deep learning model that minimize the Micro F5 score, a score that weights recall 5 times as import as precision. This means that we do not want our model to miss out on potential PII. We want fewer false negatives, but we can have slightly higher false positives in the labelling.

#### Data Sources
Learning Agency PII Dataset from Kaggle Competition

S&P 500 companies database from Wikipedia (https://github.com/datasets/s-and-p-500-companies/tree/main/data)

#### Methodology
First, we test the performance of Microsoft Presidio Analyzer combined with some custom entity recognizers and regex filtering to form a baseline Micro F5 score for recognizing PII. 

Then we use this baseline to gauge deep learning neural networks with attention layer transformers trained with variations in the number of transformer heads, number of nodes in each neural network layer, whether the model contains a Long Short-Term Memory (LSTM) layer or a traditional neural network layer, and whether the LSTM is bidirectional.

We experimented with varius inputs from sentences to the full text. Since the PII dataset is heavily imbalanced, we decide to train and validate the models on sentences. We determine which sentences have PII and create samples of datasets that have higher number of positive labels. For models with a validation score higher than 0.7, we test them on a sample of full essays with very few positive labels to see how well they would perform. We ran each combination of hyperparameters several times to account for differences in results due to gradient descent. We used the sparse categorical entropy loss function to evaluate the data, since we did not use one hot encoding on our labels.

To unpack some of this terminology above, here is a summary to describe an LSTM:

Recurrent Neural Networks (RNNs): Imagine reading a sentence where the context matters. For example, when you read “Jef is thirsty, so he orders a beer,” you connect the first part (“Jef is thirsty”) to the second part (“he orders a beer”). RNNs try to learn these logical correlations in language by remembering context. However, they struggle with long-term context.

Long Short-Term Memory (LSTM): LSTMs are a type of RNN that have memory. They decide what information to keep in mind at each step. When an LSTM encounters words like “long” and “bike,” it decides whether to remember both or forget some things. LSTMs help address the vanishing gradient problem that RNNs face, allowing them to learn better over time.

Bidirectional LSTM (BiLSTM): Now, imagine that smart reader not only reads left to right but also right to left. BiLSTM does exactly that! Unlike standard LSTM, the input flows in both directions, allowing it to utilize information from both sides. It’s like having two readers—one starting from the beginning and the other from the end—meeting in the middle to understand the whole story. BiLSTM is especially useful for modeling the relationships between words and phrases in both directions of a sequence.

#### Results
Overall, Presidio Analyzer enhanced with the custom regex, id, address, and signal phrase pattern recognizers performed quite well with a micro F5 score of 0.77. This high score is most likely due to its high recall, labelling 83% of actual positive labels correctly.

Compared to Presidio Analyzer, deep learning neural networks are more unpredictable due to gradient descent but can be trained to perform almost as well on this educational dataset. The best model architecture was the a medium-sized bidirectional LSTM network (with about 32 or 48 embed_dim and 64 or 96 ff_dim and 2 or 4 transformer heads). Bidirectional LSTMs were more reliable with less standard deviation in larger architectures than regular LSTMs. LSTMs are generally more precise than traditional neural networks, and smaller LSTMs are able to achieve better results than larger traditional neural networks. 

#### Next steps
This report focused mostly on Presidio Analyzer and LSTMs. While Presidio Analyzer was tested on the full text, we found that LSTMs performed best when trained at the sentence level. While some architectures of the LSTMs were able to perform almost as well as Presidio Analyzer, we found that performance of those models on the full text essays unsurprisingly decreased. Further research could involve determining how to transform models that perform well at the sentence level to performing well at a macro scale. 

Further research could also be done to evaluate the performance of large, pre-trained language models such as BERT on this dataset.

#### Outline of project

- [Exploratory Data Analysis](https://github.com/natalieyeo/PII-detection-education-kaggle/blob/main/Exploratory%20Data%20Analysis.ipynb)
- [Baseline Presidio Analyzer Analysis](https://github.com/natalieyeo/PII-detection-education-kaggle/blob/main/Presidio-Analyzer.ipynb)
- [Results](https://github.com/natalieyeo/PII-detection-education-kaggle/blob/main/Results.ipynb)
- Appendix:
- [Custom NLP Functions](https://github.com/natalieyeo/PII-detection-education-kaggle/blob/main/nlp-functions.ipynb)
- [Results from Model Runs](https://github.com/natalieyeo/PII-detection-education-kaggle/blob/main/nlp-models.ipynb)
