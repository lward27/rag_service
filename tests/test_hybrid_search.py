import unittest
from unittest.mock import Mock, patch

from app import main


class HybridSearchTests(unittest.TestCase):
    def test_hybrid_search_uses_fusion_query_for_rrf(self):
        response = object()
        client = Mock()
        client.query_points.return_value = response
        dense_query = [0.01] * 768

        with patch.object(main, "qdrant", client):
            result = main._hybrid_search(
                collection_name="test-collection",
                query_dense=dense_query,
                query_sparse_indices=[1, 3],
                query_sparse_values=[0.5, 0.25],
                query_filter=None,
                limit=5,
                score_threshold=0.0,
            )

        self.assertIs(result, response)
        kwargs = client.query_points.call_args.kwargs
        self.assertEqual(kwargs["collection_name"], "test-collection")
        self.assertEqual(kwargs["query"], main.FusionQuery(fusion=main.Fusion.RRF))
        self.assertEqual(len(kwargs["prefetch"]), 2)
        self.assertEqual(kwargs["prefetch"][0].using, main.DENSE_VECTOR_NAME)
        self.assertEqual(kwargs["prefetch"][0].query, dense_query)
        self.assertEqual(kwargs["prefetch"][1].using, main.SPARSE_VECTOR_NAME)
        self.assertEqual(kwargs["limit"], 5)


if __name__ == "__main__":
    unittest.main()
