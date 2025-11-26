import os
import random
import string

class DatasetGenerator:
    """Generate synthetic datasets of varying sizes for testing"""

    @staticmethod
    def create_datasets():
        """Create datasets with realistic vocabulary differences"""
        datasets = {
            'small': (10, 200),       # 10 files, ~200 unique words/file
            'medium': (100, 1000),    # 100 files, ~1000 unique words/file
            'large': (500, 500)     # 1000 files, ~5000 unique words/file
        }

        for size, (num_files, vocab_per_file) in datasets.items():
            folder = f"src/datasets/test/{size}"
            os.makedirs(folder, exist_ok=True)
            
            for i in range(num_files):
                filepath = f"{folder}/file_{i}.txt"
                if not os.path.exists(filepath):
                    with open(filepath, 'w') as f:
                        f.write(DatasetGenerator._generate_sample_text(i, vocab_per_file))


    @staticmethod
    def _generate_sample_text(file_id: int, vocab_size: int) -> str:
        """
        Generate a mix of shared & unique words for realistic TF-IDF behavior.
        - shared words appear across many documents (frequent)
        - unique words appear in a single document (rare)
        """
        random.seed(file_id)

        # Frequent global vocabulary (realistic TF, helps TF-IDF work correctly)
        global_common_words = [
            "analysis", "python", "data", "processing", "performance",
            "optimization", "machine", "learning", "model", "parallel"
        ] * 200  # repeat to increase frequency

        # Medium-frequency topic words varying across documents
        topic_seed = file_id % 5  # topic group changes every 5 docs
        topic_words = [
            f"topic_{topic_seed}_{DatasetGenerator._random_suffix()}" 
            for _ in range(vocab_size // 2)
        ]

        # Unique low-frequency words per doc (affecting IDF strongly)
        unique_words = [
            f"unique_{file_id}_{j}_{DatasetGenerator._random_suffix()}" 
            for j in range(vocab_size // 2)
        ]

        # Combine and shuffle
        content = global_common_words + topic_words + unique_words
        random.shuffle(content)

        return " ".join(content)


    @staticmethod
    def _random_suffix(length=6):
        """Random word suffix used for variation"""
        return ''.join(random.choices(string.ascii_lowercase, k=length))


# Run generator manually if needed
if __name__ == "__main__":
    DatasetGenerator.create_datasets()
