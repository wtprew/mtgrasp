def get_network(network_name):
    network_name = network_name.lower()
    if network_name == 'ggcnn':
        from .ggcnn import GGCNN
        return GGCNN
    elif network_name == 'ggcnn2':
        from .ggcnn2 import GGCNN2
        return GGCNN2
    elif network_name == 'mtgcnn':
        from .mtgcnn import MTGCNN
        return MTGCNN
    elif network_name == 'mtgcnn2':
        from .mtgcnn2 import MTGCNN2
        return MTGCNN2
    elif network_name == 'mtgcnnb':
        from .mtgcnnbranch import MTGCNNB
        return MTGCNNB
    else:
        raise NotImplementedError('Network {} is not implemented'.format(network_name))
