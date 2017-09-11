import mnist_loader
import network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net_1 = network.Network([784, 30, 10])
net_1.SGD(training_data, 30, 10, 3.0, 'bs10_lr3.0_784x30x10.csv' ,test_data=test_data)

net_2 = network.Network([784, 128, 10])
net_2.SGD(training_data, 30, 10, 3.0, 'bs10_lr3.0_784x128x10.csv' ,test_data=test_data)

net_3 = network.Network([784, 128, 10])
net_3.SGD(training_data, 30, 10, 0.5, 'bs10_lr0.5_784x128x10.csv' ,test_data=test_data)

net_4 = network.Network([784, 128, 10])
net_4.SGD(training_data, 30, 10, 50.0, 'bs10_lr50.0_784x128x10.csv' ,test_data=test_data)

net_5 = network.Network([784, 10])
net_5.SGD(training_data, 30, 10, 3.0, 'bs10_lr3.0_784x10.csv' ,test_data=test_data)

net_6 = network.Network([784, 128, 32, 10])
net_6.SGD(training_data, 30, 10, 3.0, 'bs10_lr3.0_784x128x32x10.csv' ,test_data=test_data)

