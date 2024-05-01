import torch

class ModelManager:
    __instance = None

    @staticmethod
    def get_instance():
        if ModelManager.__instance is None:
            ModelManager()
        return ModelManager.__instance

    def __init__(self):
        if ModelManager.__instance is not None:
            raise Exception("ModelManager is a singleton class. Use get_instance() method to get its instance.")
        else:
            torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
            self.modelV2 = torch.hub.load("AK391/animegan2-pytorch:main", "generator", pretrained=True, device="cpu", progress=False)
            ModelManager.__instance = self