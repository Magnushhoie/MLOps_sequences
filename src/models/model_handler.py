import base64
import io
import os
import time

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose
from ts.torch_handler.base_handler import BaseHandler


class ESM_CNN(BaseHandler):

    image_processing = Compose(
        [
            Rescale(320),
            Normalize(),
        ]
    )

    def postprocess(self, image, output):
        pred = output[0][:, 0, :, :]
        predict = self._norm_pred(pred)
        predict = predict.squeeze()
        predict_np = predict.cpu().detach().numpy()
        mask = (predict_np * 255).astype(np.uint8)

        return [self.basic_cutout(image, mask).tobytes()]

    def load_seqs(self, data):
        seqs = []

        for row in data:
            seq = row.get("data") or row.get("body")
            seqs.append(seq)

        return seqs

    def handle(self, data, context):
        """Entry point for handler. Usually takes the data from the input request and
           returns the predicted outcome for the input.
           We change that by adding a new step to the postprocess function to already
           return the cutout.

        Args:
            data (list): The input data that needs to be made a prediction request on.
            context (Context): It is a JSON Object containing information pertaining to
                               the model artefacts parameters.

        Returns:
            list : Returns the data input with the cutout applied.
        """
        start_time = time.time()

        self.context = context
        metrics = self.context.metrics

        seqs = self.load_seqs(data)
        data_preprocess = self.preprocess(seqs)

        output = self.postprocess(seqs, output)

        return output


class MyHandler(BaseHandler):
    def preprocess(self, data):
        return super().preprocess(data)

    def inference(self, data, *args, **kwargs):
        pred = self.model(self.data)

        return super().inference(data, *args, **kwargs)

    def postprocess(self, data):
        return super().postprocess(data)
