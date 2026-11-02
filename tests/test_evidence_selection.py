import json
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

from evidence_selection import assemble_context, exhaustive_question, safe_excerpt, select_candidates, validate_citations
import pilot_chat_retrieval as pilot


class EvidenceSelectionTests(unittest.TestCase):
    def test_paragraphs_are_not_cut_halfway(self):
        text = 'General.\n\n' + 'Condicion indivisible ' * 90 + '\n\nExcepcion MYPE completa.'
        result, partial = safe_excerpt(text, 100, ['mype'])
        self.assertTrue(partial)
        self.assertIn('Excepcion MYPE completa.', result)
        self.assertNotIn('Condicion indivisible', result)
        self.assertLessEqual(len(result), 100)

    def test_oversized_indivisible_sentence_is_omitted(self):
        self.assertEqual(safe_excerpt('x' * 800, 200), ('', True))

    def test_small_text_preserves_exact_numbers(self):
        self.assertEqual(safe_excerpt('General S/ 153,000; MYPE S/ 38,250.', 100)[0],
                         'General S/ 153,000; MYPE S/ 38,250.')

    def test_mixed_roles_are_balanced(self):
        rows = [{'content': str(i), 'asset_id': 1, 'chunk_index': i, 'distance': .1,
                 'document_role': 'bases'} for i in range(8)]
        rows.append({'content': 'Resultado', 'asset_id': 2, 'chunk_index': 0,
                     'distance': .9, 'document_role': 'buena_pro'})
        result = select_candidates(rows, [], [], ['bases', 'buena_pro'])
        self.assertEqual([r['document_role'] for r in result[:2]], ['bases', 'buena_pro'])

    def test_same_chunk_in_different_assets_not_deduplicated(self):
        rows = [{'content': 'Texto', 'asset_id': a, 'chunk_index': 0} for a in (1, 2)]
        self.assertEqual(len(select_candidates(rows, [], [], ['bases'])), 2)

    def test_context_preserves_archive_member_and_real_page_range(self):
        rows = [{'asset_id': 3, 'chunk_index': 4, 'source_title': 'Archivo ZIP',
                 'page_start': 2, 'page_end': 3, 'content': '[Archivo interno: Acta.pdf]\nResultado.'}]
        context, refs, evidence, _ = assemble_context(rows, 2000, lambda _: 'resultado')
        self.assertIn('Contenedor: Archivo ZIP', context)
        self.assertEqual(refs, [{'document': 'Acta.pdf', 'page': '2-3'}])
        self.assertEqual(evidence[0]['asset_id'], 3)

    def test_budget_cannot_be_exceeded(self):
        rows = [{'source_title': 'Bases', 'page_start': i, 'page_end': i,
                 'content': 'Oracion completa. ' * 1000} for i in range(1, 10)]
        context, _, _, truncated = assemble_context(rows, 1000, lambda _: '')
        self.assertLessEqual(len(context), 1000)
        self.assertTrue(truncated)

    def test_invalid_citation_not_delivered_as_verified(self):
        text, invalid = validate_citations('Dato [Bases.pdf, p. 999].', [{'document': 'Bases.pdf', 'page': '2'}])
        self.assertEqual(invalid, 1)
        self.assertNotIn('999', text)

    def test_valid_citation_kept_and_amount_not_rewritten(self):
        text, invalid = validate_citations('S/ 98,000.00 [Acta.pdf, p. 1].', [{'document': 'Acta.pdf', 'page': '1'}])
        self.assertEqual(invalid, 0)
        self.assertIn('98,000.00', text)

    def test_unambiguous_member_alias_is_canonicalized_without_false_page_precision(self):
        text, invalid = validate_citations('Dato [Acta.pdf, p. 2].', [{'document':'0Acta.pdf','page':'1-2'}])
        self.assertEqual(invalid, 0)
        self.assertEqual(text, 'Dato [0Acta.pdf, p. 1-2].')

    def test_evidence_id_is_resolved_to_its_real_document_and_pages(self):
        text, invalid = validate_citations('Dato [E2].', [],
            [{'id':'E2','document':'Acta.pdf','page':'2-3'}])
        self.assertEqual(text, 'Dato [Acta.pdf, p. 2-3].')
        self.assertEqual(invalid, 0)

    def test_model_cannot_invent_evidence_ids(self):
        text, invalid = validate_citations('Dato [E999].', [], [])
        self.assertEqual(invalid, 1)
        self.assertNotIn('E999', text)

    def test_different_document_year_is_not_an_archive_alias(self):
        text, invalid = validate_citations('Dato [2025Bases.pdf, p. 2].',
            [{'document':'2026Bases.pdf','page':'2'}])
        self.assertEqual(invalid, 1)

    def test_ambiguous_archive_member_alias_is_not_accepted(self):
        text, invalid = validate_citations('Dato [Acta.pdf, p. 2].',
            [{'document':'0Acta.pdf','page':'1-2'}, {'document':'1Acta.pdf','page':'1-2'}])
        self.assertEqual(invalid, 1)

    def test_read_lease_rolls_back_and_returns_connection_once(self):
        connection, pool = MagicMock(), MagicMock()
        connection.closed = False
        lease = pilot._ReadLease(connection, pool)
        lease.close()
        lease.close()
        connection.rollback.assert_called_once()
        pool.putconn.assert_called_once_with(connection, close=False)

    def test_broken_read_connection_is_discarded(self):
        connection, pool = MagicMock(), MagicMock()
        connection.closed = True
        pilot._ReadLease(connection, pool).close()
        pool.putconn.assert_called_once_with(connection, close=True)

    def test_exhaustive_mode_is_explicit(self):
        self.assertTrue(exhaustive_question('Enumera TODOS LOS requisitos'))
        self.assertFalse(exhaustive_question('Cual es el monto?'))

    def test_mixed_intent_and_synonym(self):
        self.assertEqual(pilot.infer_document_roles('Quien gano y que vehiculos exigian las bases?'), ('bases', 'buena_pro'))
        self.assertEqual(pilot.infer_document_roles('Cual fue el adjudicatario?'), ('buena_pro',))
        self.assertEqual(pilot.infer_document_roles('Por que no ganaron los otros postores?'), ('buena_pro',))

    def test_generated_answer_is_not_reused_as_retrieval_evidence(self):
        history = [{'role':'user','content':'Que requisitos de experiencia piden?'},
                   {'role':'assistant','content':'Ganador inventado y monto falso'}]
        query = pilot._retrieval_query('Y la MYPE?', history, 'bases')
        self.assertIn('Que requisitos de experiencia piden?', query)
        self.assertNotIn('inventado', query)
        self.assertEqual(pilot.infer_document_role('Que sustenta ese resultado?', history), 'bases')

    def test_strict_readiness_does_not_hide_database_error(self):
        with patch.object(pilot, 'enabled', return_value=True), patch.object(pilot, '_connection', side_effect=RuntimeError):
            with self.assertRaises(pilot.PilotUnavailable):
                pilot.pilot_status('1', strict=True)

    def test_stale_material_is_not_retrieved(self):
        with patch.object(pilot, 'pilot_status', return_value={'ready': False, 'state': 'stale'}):
            with self.assertRaises(pilot.PilotUnavailable) as caught:
                pilot.retrieve_pilot('1', 'Pregunta', MagicMock())
            self.assertEqual(caught.exception.code, 'PILOT_STALE')

    def test_embedding_cache_is_bounded_and_reuses_identical_query(self):
        pilot._EMBEDDING_CACHE.clear()
        client = MagicMock()
        client.with_options.return_value.embeddings.create.return_value.data[0].embedding = [.1] * 1536
        with patch.object(pilot.time, 'monotonic', return_value=1):
            self.assertFalse(pilot._query_embedding(client, 'consulta')[1])
            self.assertTrue(pilot._query_embedding(client, 'consulta')[1])
            client.with_options.assert_called_once()
        with patch.object(pilot.time, 'monotonic', return_value=302):
            self.assertFalse(pilot._query_embedding(client, 'consulta')[1])
        pilot._EMBEDDING_CACHE.clear()

    def test_dataset_has_50_questions_and_labels_limitations(self):
        data = json.loads((Path(__file__).parents[1] / 'evals/prepared_chat_cases.json').read_text(encoding='utf-8'))
        self.assertEqual(sum(len(f['questions']) for f in data['families']), 50)
        self.assertIn('parafrasis', data['source'])


if __name__ == '__main__':
    unittest.main()
