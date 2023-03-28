INDUSTRY SPECIFIC CHATBOT

Now, this project works on the domain of AI and Data analytics and addresses the real world problem of an industry specific chatbot.
There are a thousand chatbots made till date but you won't find any running chatbot for industries, there is a dire need for chatbots to work on this domain too.
This project can help many business owners by saving their precious time and money in a way that for all the times they are disturbed again n again by clients to talk on call, this chatbot would talk on their behalf. This way, the client gets a satisfactory response wothout being held up or left in doubt and the owner saves his/her time.



Now first let's talk about building the backend or we can say backbone of this project.

So now as you know, this project uses AI so out chatbot model is self learning, we have achieved this by using Neural Networks in the training model
Firstly, we'll be creating a json file that contains the questions and responses that the model will refer to to get responses.

Next we start by building the training model, we start by using the following libraries for training the model: 
```
import random
import json
import pickle
import numpy as np
import tensorflow as tf

import nltk
from nltk.stem import WordNetLemmatizer
```

Next, we use the intents json file that contains tags and responses and preprocess it into words and classes, thereafter these classes are then dumped using pickle library, these will be the files from which the model gets the responses directly:
```
words = [lemmatizer.lemmatize(word) for word in words if word not in ignoreLetters]
words = sorted(set(words))

classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))
```

next and last step of our training model is to train the data using neural networks, for our data, we have used 3 layers of neural networks for getting the results:
```
random.shuffle(training)
training = np.array(training)

trainX = training[:, :len(words)]
trainY = training[:, len(words):]

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(trainY[0]), activation='softmax'))
```

Now once the model is trained, we need to build the chatbot file which will preprocess and clean the sentence using nltk & tensorflow's keras library, tokenize the words out of th esentence which'll provide us with the output, we'll also be using an infinite loop where responses would be generated for each question until the user types quit.