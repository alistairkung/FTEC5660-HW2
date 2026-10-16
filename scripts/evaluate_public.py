"""Observe the supplied hw2.main runner on exactly the seven original public PDFs.

Instrumentation delegates to the original build_agent/score_cvs/checkpoint methods.
The provided runner owns PDF loading, concurrency, timeouts and final CSV grading.
"""

import argparse
import csv
import sys
import time
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import hw2
from scripts.inspect_batch import _save_results


def evaluate(output):
    folder = Path('public_test')
    names = [path.name for path in hw2.cv_files(folder)]
    if names != [f'CV_{i}.pdf' for i in range(1, 8)]:
        raise ValueError('Public invocation must contain exactly the seven original CV PDFs')
    if not output.resolve().is_relative_to(Path('evaluation/results').resolve()):
        raise ValueError('Raw diagnostics belong in ignored evaluation/results/')
    records, checkpoints = {}, []
    original_score, original_checkpoint = hw2.score_cvs, hw2._write_checkpoint
    def checkpoint(scores):
        original_checkpoint(scores)
        checkpoints.append(dict(scores))
    async def score(agent, cvs):
        # Match text to filenames without passing names/labels into the agent.
        from contextvars import ContextVar
        current = ContextVar('public_cv_name')
        invoke = agent.ainvoke
        async def measured(text):
            name = current.get()
            start = time.perf_counter()
            try:
                result = await invoke(text)
                records[name]['result'] = result
                return result
            finally:
                records[name]['seconds'] = time.perf_counter() - start
        agent.ainvoke = measured
        # A tiny proxy supplies per-task context while retaining the original scorer.
        class ObservedAgent:
            async def ainvoke(self, text):
                matches = [name for name, content in cvs.items() if content == text]
                if len(matches) != 1:
                    raise ValueError('Public PDF texts must be distinct for instrumentation')
                name = matches[0]
                records[name] = {}
                token = current.set(name)
                try:
                    return await agent.ainvoke(text)
                finally:
                    current.reset(token)
        scores = await original_score(ObservedAgent(), cvs)
        for name, value in scores.items():
            records[name]['score'] = value
            records[name]['prediction'] = int(value > hw2.THRESHOLD)
            result = records[name].get('result', {})
            counts = Counter(f['status'] for f in result.get('findings', []))
            records[name]['finding_counts'] = dict(counts)
            records[name]['negative_basis'] = ('MISMATCH' if counts['MISMATCH'] else
                                               'UNKNOWN' if counts['UNKNOWN'] else 'failure_or_unresolved') if value <= 0.5 else None
        return scores
    start = time.perf_counter()
    with patch.object(hw2, 'score_cvs', score), patch.object(hw2, '_write_checkpoint', checkpoint), \
         patch.object(sys, 'argv', ['hw2.py', '--cv-folder', str(folder)]):
        exit_code = hw2.main()
    rows = list(csv.DictReader(Path('results.csv').open()))
    if len(rows) != 7 or [row['cv'] for row in rows] != names or set(rows[0]) != {'cv', 'score', 'correctness'}:
        raise AssertionError('Supplied runner did not produce exactly seven correctly formatted rows')
    for row in rows:
        if row['score'] != f"{records[row['cv']]['score']:.4f}":
            raise AssertionError('Final CSV differs from scorer output')
    payload = dict(exit_code=exit_code, seconds=time.perf_counter() - start, records=records,
                   csv_rows=rows, checkpoint_count=len(checkpoints), checkpoints=checkpoints,
                   per_cv_timeout_seconds=hw2.PER_CV_TIMEOUT_SECONDS, max_concurrency=3)
    _save_results(output, payload)
    for name, record in records.items():
        print(name, record['result'].get('selected_candidate_id') if 'result' in record else None,
              record['score'], record['negative_basis'], round(record['seconds'], 2))
    print(f'Total: {payload["seconds"]:.2f}s; checkpoints: {len(checkpoints)}; CSV rows: {len(rows)}')
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    evaluate(args.output)


if __name__ == '__main__':
    main()
