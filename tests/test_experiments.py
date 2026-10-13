from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import experiments
from server import Store, app
from test_core import DummyEncoder
from fastapi.testclient import TestClient

class ExperimentTests(unittest.TestCase):
    def test_manifest(self):
        catalog=experiments.manifest()
        self.assertEqual(len(catalog),7)
        self.assertIn('text',{e['id'] for e in catalog})
        self.assertEqual(experiments.EXPERIMENTS['text'].expected_key,'pump')

    def test_text_isolation_and_metrics(self):
        st=Store()
        st.load_experiment('text')
        self.assertEqual(len(st.experiment_docs['text']),4)
        st.model=DummyEncoder();st.state='ready';st.reindex()
        result=st.evaluate_experiment('text',768,3)
        self.assertEqual(result['corpus_size'],4)
        self.assertIn(result['rank'],(1,2,3,4))
        self.assertEqual(result['reciprocal_rank'], round(1/result['rank'],4))
        self.assertEqual(result['task'],'SearchQuery')
        self.assertEqual(len(result['results']),3)
        # Ensure seeds cannot contaminate evaluation.
        ids=set(st.experiment_docs['text'].values())
        self.assertTrue(all(r['id'] in ids for r in result['results']))
        comparison=st.compare_experiment_dimensions('text')
        self.assertEqual([r['dimension'] for r in comparison['dimensions']],[128,256,512,768])

    def test_experiment_load_is_idempotent(self):
        st=Store();st.load_experiment('text');n=len(st.docs)
        st.load_experiment('text');self.assertEqual(len(st.docs),n)

    def test_media_generation_and_pdf_extract(self):
        from pypdf import PdfReader
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)
            for typ in ('image','audio','pdf'):
                item=experiments.EXPERIMENTS[typ].files[0]
                path=experiments.create_media(typ,item,base)
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size,100)
                if typ=='pdf':self.assertIn('blocked coolant', PdfReader(path).pages[0].extract_text())

    def test_api_404_and_load_before_real(self):
        from server import store
        with TestClient(app) as client:
            self.assertEqual(client.get('/api/experiments').status_code,200)
            self.assertEqual(client.post('/api/experiments/unknown/load').status_code,404)
            self.assertEqual(client.post('/api/experiments/text/run',json={}).status_code,409)
            resp=client.post('/api/experiments/text/load')
            self.assertEqual(resp.status_code,200,resp.text)
            self.assertEqual(len(resp.json()['documents']),4)
            self.assertEqual(client.get('/api/experiments').json()['experiments'][0]['loaded'],True)

if __name__=='__main__':unittest.main()