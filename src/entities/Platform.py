import json
import os

class Platforme:
    """Execution platform"""

    # Static attribute to store available platforms
    platforms = []
    def __init__(self, type, name, memory):
        self.type = type
        self.name = name
        self.memory = memory
        Platforme.platforms.append(self)


    def __str__(self):
        return f"{self.type}_{self.name}"


    def to_json(self):
        # Convert attributes to a dictionary
        data = {
            "type": self.type,
            "name": self.name,
            "memory": self.memory,
        }
        return data


    def serialize (self):
        json_data = self.to_json()
        with open(f"./Platforms/{self.type}_{self.name}.json", "w") as file :
            json.dump(json_data, file)

    def get_platform(self, path):
        file = open(path, "r")
        data = json.loads(file.read())
        self.name = data["name"]
        self.type= data["type"]
        self.memory = int(data["memory"])
        Platforme.platforms.append(self)

    @staticmethod
    def getPlatforms():
        dir = '../output/Platforms'
        for path in os.listdir(dir):
            file = open(f"{dir}/{path}", "r")
            data = json.loads(file.read())
            platform = Platforme(data["type"],data["name"], float(data["memory"]))
            Platforme.platforms.append(platform)
        return Platforme.platforms

    @staticmethod
    def getPlatformByName(name):
        for platform in Platforme.platforms:
            if f"{platform.type} {platform.name}" == name:
                return platform
        return None
