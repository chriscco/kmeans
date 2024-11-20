import json 

conversation_path = "files/conversation_graphs.json"
proximity_path = "files/proximit_graphs.json"
attention_path = "files/shared_attention_graphs.json"

def get_convo():
    return reader(conversation_path);



def get_attention():
    return reader(attention_path);



def get_proximity():
    return reader(proximity_path);


def reader(filename): 
    # Load the JSON data from the specified file
    with open(conversation_path, 'r') as f:
        data = json.load(f)

    return data