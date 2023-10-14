from core.builders.flying_chairs_builder import FlyingChairsDataGenerator
from core.builders.input_data_builder import  InputDataBuilder
from models.flow_nets import FlowNet
from tensorflow.python.client import device_lib
if __name__ == '__main__':
    # if out of gpu memory error, try smaller batch_size
    data_generator = FlyingChairsDataGenerator(batch_size=16)
    flow_net = FlowNet()
    flow_net.create_model()
    flow_net.train(data_generator, epochs=10)
