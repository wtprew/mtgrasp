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
    elif network_name == 'mtgcnnb2':
        from .mtgcnnbranch2 import MTGCNNB2
        return MTGCNNB2
    elif network_name == 'mtgcnnb3':
        from .mtgcnnbranch3 import MTGCNNB3
        return MTGCNNB3
    elif network_name == 'sg':
        from .salgrasp import SGCNN
        return SGCNN
    elif network_name == 'sg2':
        from .salgrasp2 import SGCNN2
        return SGCNN2
    elif network_name == 'sg2_a':
        from .salgrasp2_a import SGCNN2
        return SGCNN2
    elif network_name == 'sg2_b':
        from .salgrasp2_b import SGCNN2
        return SGCNN2
    elif network_name == 'sg2_s':
        from .salgrasp2_separated import SGCNN2
        return SGCNN2
    elif network_name == 'cel':
        from .salgraspcel import SGCNN
        return SGCNN
    else:
        raise NotImplementedError('Network {} is not implemented'.format(network_name))
