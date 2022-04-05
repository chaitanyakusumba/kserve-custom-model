import kserve
import spacy
import json
from typing import Dict

class SpacyModel(kserve.KFModel):
    def __init__(self, name: str):
       super().__init__(name)
       self.name = name
       self.load()

    def load(self):
        self.model = spacy.load("NER_trained/spacy_roberta_model")
        self.ready = True

    def predict(self, request: Dict) -> Dict:
        inputs = request["instances"]
        djson = list()
        for eachInput in inputs:
            
            doc = self.model(eachInput)
            labels = list()
            for e in doc.ents:
                labels.append({'start':e.start_char,'end': e.end_char,'label': e.label_, 'text': e.text})
            djson.append({'text': doc.text, "labels": labels})
            
        return json.dumps(djson)

if __name__ == "__main__":
    model = SpacyModel("patient-emotex-ner")
    kserve.KFServer().start([model])