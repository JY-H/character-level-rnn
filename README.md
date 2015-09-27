# character-level-rnn
This script implements a multi-layer LSTM for training character-level language models. The model learns to predit the next character in a sequence. 

The language model was initially describe in [this
blogpost](http://karpathy.github.io/2015/05/21/rnn-effectiveness/),
and written in Torch. I have rewritten the model in
[Keras](https://github.com/fchollet/keras). 

## Training
- Currently the model trains on 2-layered LSTM with 512 hidden nodes, with a batch size of 100 and 20 character length. 
- After 60 iterations, the model has learned basic English phrases and puncutation, but not Shakespearean prose. I have changed the model to be 3-layered, 512 hidden nodes, batch size of 100 and 60 character length. 

<!--
## Training
So far I have only trained the model on the Shakespeare file in the
original blog post, but intend to train it on a corpus of Haikus to
see whether the model can learn to create a theme in its text
generation.  
-->

