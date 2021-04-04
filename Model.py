from datetime import datetime
import time
from threading import Thread
from random import sample
from scipy.spatial import distance

import pandas as pd
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from Status import Status


class Model:

    # Timeout for training thread in seconds
    timeout = 120

    def __init__(self, objects, l, k, callback):

        self.objects = objects
        self.l = l
        self.k = k
        self.callback = callback

        self.status = Status.NEW
        self.models = []
        self._exceptions = []
        self._generate_thread = None

        self.start_time = None
        self.finish_time = None

    """
    Start model generation n separate thread
    """
    def start(self):
        self._generate_thread = Thread(target=self._generate_models, args=(), daemon=True)
        self._generate_thread.start()

    """
    Predict representation score for given objects
    """
    def predict(self, objects):

        # Make predictions only when model ready
        if len(self.models) < 1 or self.status != Status.READY:
            return
        
        df = pd.DataFrame()
        # Get prediction for each model
        for i, model in enumerate(self.models):
            # Save predictions for each model
            df[i] = model.predict(objects)
        # Return mean values for each row across the models
        return [np.mean(row) for index, row in df.iterrows()]

    """
    Return all exceptions in form of the object
    """
    def get_exceptions(self):
        exceptions = []
        for exception in self._exceptions:
            exceptions.append({
                "time": str(exception[0]),
                "description": str(exception[1])
             })
        return exceptions

    """ 
    Add exception to exceptions list and change status to FAULT
    """
    def _add_exception(self, e):
        self._exceptions.append((datetime.now(), e))
        self.status = Status.FAULT

    """ 
    Divide objects randomly into L buckets in similar size
    """
    def _divide_to_buckets(self):

        # Get minimal number of items in one bucket
        size = int(len(self.objects) / self.l)
        # Get number of items that could not be divided evenly
        remainder = len(self.objects) % self.l

        objects = self.objects

        # For each bucket
        for i in range(self.l):
            total_size = size
            # Check if number of items in bucket should be increased due to remainder
            if remainder > 0:
                total_size = size + 1
                remainder -= 1
            # Get random objects for bucket
            random_objects = sample(objects, total_size)
            # Remove selected objects from all objects
            for random_object in random_objects:
                objects.remove(random_object)
            yield random_objects

    """  
    Generate model for each bucket and save it in model list
    """
    def _generate_model(self, bucket):

        try:

            # Array for samples
            X = []
            # Array for labels
            y = []

            for object in bucket:
                # Add object to training set
                X.append(object)
                object_distance = []
                # Add distance from the object to every other object in the bucket to array
                for object_other in bucket:
                    if object is not object_other:
                        object_distance.append(distance.euclidean(tuple(object), tuple(object_other)))

                # Take k lowest distances if applicable
                if self.k is not None:
                    # Sort distances ascending
                    object_distance.sort()
                    object_distance = object_distance[:self.k]

                # Calculate representation
                representation = 1 / (1 + np.mean(object_distance))
                # Add representation to array with labels
                y.append(representation)

            # Create classifier
            model = make_pipeline(PolynomialFeatures(4), Ridge())
            # Train classifier
            model.fit(X, y)
            # Add classifier to array with models
            self.models.append(model)

        except Exception as e:
            self._add_exception(e)

    """   
    Generate all the models in different threads
    """
    def _generate_models(self):

        # Set start time
        self.start_time = datetime.now()
        self.status = Status.TRAINING

        try:
            threads = []

            # Divide all objects into l buckets
            for bucket in self._divide_to_buckets():
                # For each bucket start model generation in new thread
                x = Thread(target=self._generate_model, args=(bucket,), daemon=True)
                x.start()
                threads.append(x)

            # Wait for threads to finish
            for thread in threads:
                thread.join(timeout=self.timeout)
                # Check if thread timed out
                if thread.is_alive():
                    self._add_exception(Exception("Timeout"))

        except Exception as e:
            self._add_exception(e)

        # Set finish time
        self.finish_time = datetime.now()
        # Change status to "READY" if no faults
        if self.status != Status.FAULT:
            self.status = Status.READY

        # Invoke callback function when generation finished
        self.callback(self)
