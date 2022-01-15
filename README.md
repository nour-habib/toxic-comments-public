<h1>Toxic Comment Classification</h1>

This is a group project for a 4th year level course in Computational Linguistics.

<h2> Introduction </h2>

This project evaluates the effectiveness of mul- tiple methods on multi-label classification of toxicity in online comments. We compare the accuracy, precision, recall and F1-scores of Recurrent Neural Networks (RNN), Long-Short- Term-Memory Networks (LSTM), Gated Re- curring Unit Networks (GRU), Convolutional Neural Networks (CNN), Fully Connected Neural Networks (FCN), and the Naive Bayes Classifier. The RNN based Neural Networks were trained as both bi-directional and uni-directional, and each Neural Net- work was trained with and without the GloVe word embeddings. We evaluated our methods on Wikipedia comments sourced by the Kaggle Toxic Comment Classification Challenge dataset. Our results found that the bi-directional and uni-directional LSTM net- works offered the best F1-score, and therefore is the most effective at multi-label classification. There was no perceived difference in performance between uni-directional and bi- directional LSTM Networks. In the future, we seek to widen our testing by implementing a variety of word embeddings and utilizing deeper and more complex network architectures.

This project tests different machine learning models to classify any type of abusive, offensive comments. The training data was taken from Wikipedia's comment section and manually tagged for toxicity. The various models that were tested are LSTM, Naive Bayes, CNN, RNN and GRU. LSTM was the best performing model for classifying toxic comments.

<h2> Approach </h2>

This project studies the performance of mul- tiple Neural Network models (LSTM, RNN, GRU, CNN, FCN) on multi-label classification. The RNN based neural networks (LSTM, RNN, GRU) are compared using bi-directional and uni- directional models. Each neural network is trained and compared with and without word level GloVe embeddings. Additionally, we also implement a naive bayes classifier to determine if statistical clas- sification methods are still viable today in a field slowly being overtaken by machine learning. Our approaches are detailed below.

<h3> Multi-Label Classification </h3> 

The goal for mutli-label classification is to determine if a given comment is toxic or non-toxic. If the comment is classified as toxic, we can potentially further categorize the comment into 5 additional categories:
• Severely Toxic: Extremely hateful comments. Usage of slurs, derogatory and offensive words.
• Obscene: Offensive or disgusting statements by accepted standards of morality and de- cency, usually in the portrayal of sexual mat- ters
• Threat: A statement with the intention to in- flict injury, death, or other hostile action on another person
• Insult: An expression or statement that is dis- respectful or scornful
• Identity hate: Hatred, hostility, or violence towards members of a race, ethnicity, nation, religion, gender, gender identity, sexual orien- tation or any other designated sector of society
A comment may be categorized into multiple categories. For example, a particular toxic com- ment may be further classified as a threat and an obscene comment.

<h2> Results </h2>

<img src="https://github.com/nour-habib/toxic-comments-public/blob/main/images/accuracy.png">
<img src="https://github.com/nour-habib/toxic-comments-public/blob/main/images/f1.png">
<img src="https://github.com/nour-habib/toxic-comments-public/blob/main/images/percision.png" >
<img src="https://github.com/nour-habib/toxic-comments-public/blob/main/images/recall.png">
