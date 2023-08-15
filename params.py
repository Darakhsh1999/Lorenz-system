
class Triplets():
    """ Container class for parameters triplets (sigma, rho, beta) """

    def __init__(self):
        self.triplets = {
            "chaos": [],
        }
    
    def get_params(self, mode="chaos"):
        return self.triplets.get(mode)
