def initiate_model(architecture, in_channel, out_channel):
    model = architecture(in_channel, out_channel)  
    return model