import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server import Store, chunks, unit

class CoreTests(unittest.TestCase):
    def test_chunking_covers_small_doc(self):
        self.assertEqual(chunks(" hello  world "), ["hello world"])
        self.assertEqual(chunks("  "), [])

    def test_chunking_large_doc(self):
        text = "nearby geology lithology prediction " * 90
        result = chunks(text)
        self.assertGreater(len(result), 1)
        self.assertTrue(all(len(s) < 1000 for s in result))

    def test_truncation_renormalizes(self):
        vector = np.ones(768)
        v = unit(vector, 128)
        self.assertEqual(v.shape, (128,))
        self.assertAlmostEqual(float(np.dot(v, v)), 1.0, places=6)

    def test_invalid_dimension(self):
        with self.assertRaises(ValueError):
            unit(np.ones(768), 42)

    def test_store_in_memory(self):
        store = Store()
        n = len(store.docs)
        doc = store.add_text("Sample", "Text about information retrieval.")
        self.assertEqual(len(store.docs), n+1)
        self.assertTrue(store.delete(doc.id))
        self.assertEqual(len(store.docs), n)

if __name__ == "__main__":
    unittest.main()

# The tests below exercise retrieval logic through a deterministic fake encoder;
# they are NOT a substitute for testing the actual checkpoint after download.
class DummyEncoder:
    def encode(self, value, **kwargs):
        import hashlib, re
        def embed(text):
            v = np.zeros(768, dtype=np.float32)
            for word in re.findall(r'\w+', str(text).lower()):
                index = int.from_bytes(hashlib.sha256(word.encode()).digest()[:4], 'big') % 128
                v[index] += 1
            if np.linalg.norm(v) == 0:
                v[0] = 1
            return v / np.linalg.norm(v)
        if isinstance(value, list):
            return np.stack([embed(x) for x in value])
        return embed(value)


class FakeEncoderTests(unittest.TestCase):
    def test_search_returns_ranked_sources(self):
        store = Store()
        store.model = DummyEncoder()
        store.state = 'ready'
        store.reindex()
        result = store.search('embedding vector search', 256, 4)
        self.assertEqual(result['mode'], 'real')  # API shape; encoder here is fake.
        self.assertEqual(len(result['vector_preview']), 16)
        self.assertLessEqual(len(result['results']), 4)
        self.assertTrue(all(-1.0001 <= r['score'] <= 1.0001 for r in result['results']))
        self.assertTrue(all(result['results'][i]['score'] >= result['results'][i+1]['score'] for i in range(len(result['results'])-1)))

    def test_fake_similarity_and_dim(self):
        store = Store()
        store.model = DummyEncoder()
        store.state = 'ready'
        result = store.similarity('same words', 'same words', 128)
        self.assertEqual(result['dimension'], 128)
        self.assertAlmostEqual(result['score'], 1.0, places=4)
