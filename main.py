from core.builders import InputDataBuilder
from models.flow_nets import FlowNet
if __name__ == '__main__':
    input_data_builder = InputDataBuilder().build()
    flow_net = FlowNet()
    flow_net.create_model()
    #flow_net.model.
