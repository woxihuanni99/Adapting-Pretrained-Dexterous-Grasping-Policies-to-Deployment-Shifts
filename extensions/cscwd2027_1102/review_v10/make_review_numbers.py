"""Review of existing correction results and repeat-evaluation references.

No new test family, model selection, training or simulator evaluation.
"""
from pathlib import Path
import hashlib
import json

R = Path(__file__).resolve().parent
B = R.parent
SP = R.parent / 'review_v5/numbers.json'
NP = R.parents[2] / 'analysis/numbers.json'
XP = R.parent / 'results/numbers.json'
S = json.loads(SP.read_text(encoding='utf8'))
N = json.loads(NP.read_text(encoding='utf8'))
X = json.loads(XP.read_text(encoding='utf8'))
records = []
for name, label in [('Residual network', 'Residual GRPO'), ('Direct constant + scale', 'Constant + scale (48)'), ('Scaled shift', 'Scaled shift (24)')]:
    r = next(x for x in S['constant_diagnostics'] if x['name'] == name)
    records.append(dict(batch='Initial', label=label, parameters=r['parameters'], success=r['success'], loss=r['loss'], mean_le_5=r['loss']['mean'] <= 5, upper_CI_le_5=r['loss']['CI95'][1] <= 5))
r = next(x for x in S['same_runtime_selected'] if x['method'] == 'res_grpo' and x['condition'] == 'LF')
records.append(dict(batch='Repeat', label='Residual GRPO', parameters='state-dependent', success=r['success'], loss=r['loss'], mean_le_5=r['loss']['mean'] <= 5, upper_CI_le_5=r['loss']['CI95'][1] <= 5))
old_frozen = 100 * N['frozen']['nom']
new_frozen = X['frozen_test_success_pct']['nom']
old_loss = records[0]['loss']['mean']
new_loss = records[-1]['loss']['mean']
out = dict(sources=[dict(path=p.relative_to(R.parents[2]).as_posix(), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in [SP, NP, XP]],
           record_rows=records,
           repeat_evaluation=dict(initial_frozen_nominal_pct=old_frozen, repeat_frozen_nominal_pct=new_frozen, frozen_change_pp=new_frozen-old_frozen, initial_resGRPO_loss_pp=old_loss, repeat_resGRPO_loss_pp=new_loss, loss_change_pp=new_loss-old_loss, implied_adapted_nominal_change_pp=(new_frozen-new_loss)-(old_frozen-old_loss)),
           interpretation='Keep each loss with its own evaluation reference. Report threshold sensitivity without declaring a robust pass/fail ranking across batches. Compact controls have not been evaluated in the repeat batch.',
           new_training=0, new_simulation=0)
(R / 'numbers.json').write_bytes(json.dumps(out, ensure_ascii=False, indent=2).encode('utf8'))
print(json.dumps(out['repeat_evaluation'], ensure_ascii=False, indent=2))
