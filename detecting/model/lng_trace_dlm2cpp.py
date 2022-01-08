import torch
import os

class TextClassificationModel(torch.nn.Module):

    def __init__(self):
        super(TextClassificationModel, self).__init__()
        self.fc1 = torch.nn.Linear(100, 250)
        self.fc2 = torch.nn.Linear(250, 50)
        self.fc3 = torch.nn.Linear(50, 2)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc1.bias.data.zero_()
        self.fc2.weight.data.uniform_(-initrange, initrange)
        self.fc2.bias.data.zero_()
        self.fc3.weight.data.uniform_(-initrange, initrange)
        self.fc3.bias.data.zero_()

    def forward(self, vec_text):
        x = torch.nn.functional.relu(self.fc1(vec_text))
        x = torch.nn.functional.relu(self.fc2(x))
        return torch.nn.functional.softmax(self.fc3(x))

class PornTextDetector(torch.nn.Module):

    def __init__(self) -> None:

        super(PornTextDetector, self).__init__()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.porn_detector_model = torch.load('./PornTextDetector_2.pth',
                                    map_location=device)

        self.porn_detector_model.eval()

        self.porn_detector_model = torch.jit.trace(self.porn_detector_model,
                            torch.rand((1, 100),
                            dtype=torch.float))

    def forward(self, vectorize_text):

        return self.porn_detector_model(vectorize_text)

if __name__ == "__main__":

    module = PornTextDetector()
    sm = torch.jit.script(module)

    sm.save('annotation_porn_text_detector.pt')
