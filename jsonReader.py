import json 


class JsonRead:
    __conversation_path = "files/conversation_graphs.json"
    __proximity_path = "files/proximity_graphs.json"
    __attention_path = "files/shared_attention_graphs.json"

    __conversation_data = [];
    __attention_data = []
    __proxi_data = []

    def reader(self, filename): 
        # Load the JSON data from the specified file
        with open(filename, 'r') as f:
            data = json.load(f)

        if filename == self.__conversation_path:
            self.__conversation_data = data["graphs"]
        elif filename == self.__attention_path:
            self.__attention_data = data["graphs"]
        else: self.__proxi_data = data["graphs"]

        return data["graphs"]

    def get_conversation(self):
        if self.__conversation_path != []:
            return self.__conversation_data
        else: return self.reader(self.__conversation_path)

    def get_attention(self):
        if self.__attention_data != []:
            return self.__attention_data
        else: return self.reader(self.__attention_path);

    def get_proximity(self):
        if self.__proxi_data != []:
            return self.__proxi_data
        else: return self.reader(self.__proximity_path);


if __name__ == "__main__":
    j = JsonRead()
    print(j.get_proximity())