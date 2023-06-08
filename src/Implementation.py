import json
import os
from IDSModel import IDSModel
from Platform import Platforme

class Implementation:

    """ Characteristics of an IDS Model on a given platform """
    # Static attribute to store existing implementations
    implementations = []

    def __init__(self, model, platforme, characterization_volume, accuracy, f1_score, inference_time,  energy, peak_memory, model_size, description ):
        # The IDS to be characterized
        self.model = model

        # The platform where the characterization has been performed
        self.platforme = platforme

        # Volume of data on which the characterization has been performed
        self.characterization_volume = characterization_volume


        # QoS metrics
        self.accuracy = accuracy
        self.f1_score = f1_score
        self.inference_time = inference_time
        self.energy = energy

        # Resource metrics
        self.peak_memory = peak_memory
        self.model_size = model_size
        self.description = description
        Implementation.implementations.append(self)

    def __str__(self):
        return f"Implementation of: {self.model.name} on {self.platforme.type} {self.platforme.name} \n\tAccuracy : {self.accuracy}\n\tF1-Score : {self.f1_score}\n" \
               f"\tInference time of {self.characterization_volume} observations : {self.inference_time} ms => {self.inference_time/self.characterization_volume} ms per observation \n" \
               f"\tEnergy of inference on  {self.characterization_volume} : {self.energy} J => {self.energy/self.characterization_volume} J per observation\n" \
               f"\tPeak memory : {self.peak_memory}\n"\
               f"\tModel size : {self.model_size}\n"

    def to_json(self):
        # Convert attributes to a dictionary
        data = {
            "model": self.model.name,
            "platforme": f"{self.platforme.type} {self.platforme.name}",
            "characterization_volume": self.characterization_volume,
            "accuracy": self.accuracy,
            "f1_score": self.f1_score,
            "inference_time": self.inference_time,
            "peak_memory": self.peak_memory,
            "model_size": self.model_size,
            "energy": self.energy,
            "description": self.description,
        }
        return data


    def serialize (self):
        json_data = self.to_json()
        with open(f"../output/Implementations/{self.model.name}_{self.platforme}.json", "w") as file :
            json.dump(json_data, file)


    @staticmethod
    def getImplementations():
        dir = '../output/Implementations'
        for path in os.listdir(dir):
            file = open(f"{dir}/{path}", "r")
            data = json.loads(file.read())
            model = IDSModel(data['model'],None, None)
            platforme = Platforme.getPlatformByName(data['platforme'])
            impl = Implementation(model,platforme, int(data["characterization_volume"]), float(data["accuracy"]), float(data["f1_score"]), float(data["inference_time"]), float(data["energy"]), None,None, None)
            Implementation.implementations.append((impl))
        return Implementation.implementations

    @staticmethod
    def getImplementationsDict():
        implemDict = []
        dir = '../output/Implementations'
        for path in os.listdir(dir):
            file = open(f"{dir}/{path}", "r")
            implemDict.append(json.loads(file.read()))

        return implemDict

