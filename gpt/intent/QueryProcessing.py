import aiofiles
import asyncio
import yaml
import numpy as np
from sentence_transformers import SentenceTransformer

class QueryProcessor:
    @staticmethod
    async def load_yaml(filename: str) -> dict:
        """
        Load a YAML file and return its contents as a dictionary.

        Args:
            filename (str): The path to the YAML file.

        Returns:
            dict: The contents of the YAML file as a dictionary.
        """
        async with aiofiles.open(filename, 'r') as file:
            data = await file.read()
        return yaml.safe_load(data)

    @staticmethod
    async def get_homes(folder: str, num: int) -> list:
        """
        Load multiple YAML files from a folder and return their contents as a list.

        Args:
            folder (str): The path to the folder containing the YAML files.
            num (int): The number of YAML files to load.

        Returns:
            list: A list of dictionaries, where each dictionary contains the contents of a YAML file.
        """
        return await asyncio.gather(*[QueryProcessor.load_yaml(folder + f"home{i+1}.yaml") for i in range(num)])

    @staticmethod
    def vectorize(data: list) -> np.ndarray:
        """
        Vectorize a list of sentences using SentenceTransformer.

        Args:
            data (list): A list of sentences to vectorize.

        Returns:
            np.ndarray: A numpy array containing the vector representations of the input sentences.
        """
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        return model.encode(data, convert_to_numpy=True)

    @staticmethod
    def similarity_search(user_vector: np.ndarray, digi_vectors: np.ndarray, num: int) -> np.ndarray:
        """
        Perform a similarity search to identify the closest vectors to the user's query.

        Args:
            user_vector (np.ndarray): The vector representation of the user's query.
            digi_vectors (np.ndarray): A numpy array containing the vector representations of the available options.
            num (int): The number of closest options to return.

        Returns:
            np.ndarray: A numpy array containing the indices of the closest options.
        """
        distances = np.dot(digi_vectors, user_vector) / (np.linalg.norm(digi_vectors, axis=1) * np.linalg.norm(user_vector))
        top_indices = distances.argsort()[-num:][::-1]
        return top_indices

    @staticmethod
    def extract_names_from_mount(mount: dict) -> list:
        """
        Recursively extract names from mount sections.

        Args:
            mount (dict): A dictionary containing mount sections.

        Returns:
            list: A list of names extracted from the mount sections.
        """
        names = []
        for value in mount.values():
            if isinstance(value, dict):
                if 'name' in value:
                    names.append(value['name'])
                else:
                    names.extend(QueryProcessor.extract_names_from_mount(value))
        return names
