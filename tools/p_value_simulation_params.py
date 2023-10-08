class PValueSimulationParams(object):
    def __init__(self, x_sample_size, y_sample_size, x_p, y_p, iter_size):
        self.x_sample_size = x_sample_size
        self.y_sample_size = y_sample_size
        self.x_p = x_p
        self.y_p = y_p
        self.iter_size = iter_size
        self.sample_name = f"x_p={x_p}, y_p={y_p}, x_sample_size={x_sample_size}, y_sample_size={y_sample_size}"
