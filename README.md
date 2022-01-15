<h1>Toxic Comment Classification</h1>

This is a group project for a 4th year level course in Computational Linguistics, Fall 2020.
Group members: Nathan Cheung, Gregory Dewar, Fan-Yu Meng, Nour Habib

<h2> Introduction </h2>

This project evaluates the effectiveness of multiple methods on multi-label classification of toxicity in online comments. We compare the accuracy, precision, recall and F1-scores of **Recurrent Neural Networks (RNN)**, **Long Short-Term Memory Networks (LSTM)**, **Gated Recurring Unit Networks (GRU)**, **Convolutional Neural Networks (CNN)**, **Fully Connected Neural Networks (FCN)**, and the **Naive Bayes Classifier**. The RNN based Neural Networks were trained as both bi-directional and uni-directional, and each Neural Network was trained with and without the GloVe word embeddings. We evaluated our methods on Wikipedia comments sourced by the Kaggle Toxic Comment Classification Challenge dataset. Our results found that the bi-directional and uni-directional LSTM networks offered the best F1-score, and therefore is the most effective at multi-label classification. There was no perceived difference in performance between uni-directional and bi- directional LSTM Networks.

<h2> Approach </h2>

This project studies the performance of multiple Neural Network models (LSTM, RNN, GRU, CNN, FCN) on multi-label classification. The RNN based neural networks (LSTM, RNN, GRU) are compared using bi-directional and uni- directional models. Each neural network is trained and compared with and without word level GloVe embeddings. Additionally, we also implement a Naive Bayes classifier to determine if statistical classification methods are still viable today in a field slowly being overtaken by machine learning. Our approaches are detailed below.

<h3> Multi-Label Classification </h3> 

The goal for multi-label classification is to determine if a given comment is toxic or non-toxic. If the comment is classified as toxic, we can potentially further categorize the comment into 5 additional categories:<br>

• **Severely Toxic**: Extremely hateful comments. Usage of slurs, derogatory and offensive words.<br>
• **Obscene**: Offensive or disgusting statements by accepted standards of morality and decency, usually in the portrayal of sexual matters<br>
• **Threat**: A statement with the intention to inflict injury, death, or other hostile action on another person<br>
• **Insult**: An expression or statement that is disrespectful or scornful<br>
• **Identity hate**: Hatred, hostility, or violence towards members of a race, ethnicity, nation, religion, gender, gender identity, sexual orientation or any other designated sector of society<br>

A comment may be categorized into multiple categories. For example, a particular toxic comment may be further classified as a threat and an obscene comment.

<h2> Results </h2>

<img src="https://github.com/nour-habib/toxic-comments-public/blob/main/images/accuracy.png">
<img src="https://github.com/nour-habib/toxic-comments-public/blob/main/images/f1.png">
<img src="https://github.com/nour-habib/toxic-comments-public/blob/main/images/percision.png" >
<img src="https://github.com/nour-habib/toxic-comments-public/blob/main/images/recall.png">
