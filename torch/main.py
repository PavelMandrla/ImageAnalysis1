from Neural_Network import Neural_Network
import torch
import torch.nn as nn

train_data = (  (0.111052, 0.071038), (0.155572, 0.068847), (0.150963, 0.069233), (0.153282, 0.070089),
                (0.708833, 0.070819), (0.761471, 0.067892), (0.725034, 0.071210), (0.707275, 0.073751),
                (0.166867, 0.834633), (0.219282, 0.842084), (0.219144, 0.874569), (0.219657, 0.842442))


train_labels = ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), # # - square
              (0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0),   # * - star
              (0.0, 0.0, 1.0), (0.0, 0.0, 1.0), (0.0, 0.0, 1.0), (0.0, 0.0, 1.0),)  # ## - rectangle

# fill your own feature vectors from test images
test_data = ((0.115522, 0.067124), #square
            (0.209803, 0.833025),  #rect
            (0.689388, 0.069393),  #star
            (0.152290, 0.066667),  #square
            (0.120417, 0.893520),  #rect
            (0.159294, 0.899029),  #rect
            (0.731309, 0.072402),  #star
            (0.746990, 0.069098),  #star
            (0.135408, 0.066944))  #square


X = torch.tensor(train_data,    dtype=torch.float)
y = torch.tensor(train_labels,  dtype=torch.float)
xPredicted = torch.tensor(test_data, dtype=torch.float)


NN = Neural_Network(2, 3, 5)

iter = 0
loss = NN.train(X, y)
while loss > 0.005:
    loss = NN.train(X, y)
    iter += 1

print("Loss %f (%d iterations)" % (loss, iter))
print("Finished training!")

prediction = NN.predict(xPredicted)
names = ["square", "star", "rectangle"]
for i in range(len(prediction)):
    print(str(test_data[i]) + " -> " + names[prediction[i]])
