#!/usr/bin/env python3
"""Independent standard-library validation of 18 Aer comparisons and Stage 2/3."""
import csv,json,math,itertools
from pathlib import Path
R=Path(__file__).resolve().parent
E=R/'experiments/p3_p8/results'
read=lambda p:json.loads(p.read_text())
s1=list(csv.DictReader((R/'stage1_result.csv').open()))
assert len(s1)==100 and len(set(r['customer_id'] for r in s1))==100
ref=read(E/'exact_reference.json')
ids=ref['customer_ids_q0_to_q9'];net=ref['net_values'];cost=ref['costs'];cap=ref['capacities']
assert ids==[r['customer_id'] for r in s1[:10]]
poss=[]
for ix in itertools.combinations(range(10),6):
 c=sum(cost[i] for i in ix);h=sum(cap[i] for i in ix);v=sum(net[i] for i in ix)
 if c<=25 and h<=22:poss.append((v,ix))
assert len(poss)==210
v,ix=max(poss)
assert math.isclose(v,ref['exact_optimum'],abs_tol=1e-9)
assert ''.join('1' if i in ix else '0' for i in reversed(range(10)))==ref['optimal_bitstring_q9_to_q0']
rows=read(E/'comparison_p3_p8.json');assert len(rows)==18
for row in rows:
 m=row['model'];p=int(row['depth_p']);assert 3<=p<=8
 counts=read(E/f'{m}_p{p}_aer_counts.json');assert sum(counts.values())==8192,(m,p)
 assert all(len(bit)==10 and set(bit)<={'0','1'} and n>=0 for bit,n in counts.items())
 nfeas=sum(n for bit,n in counts.items() if bit.count('1')==6 and sum(cost[j] for j in range(10) if bit[-1-j]=='1')<=25 and sum(cap[j] for j in range(10) if bit[-1-j]=='1')<=22)
 assert nfeas==row['aer_feasible_shots'],(m,p,'feasibility')
 assert counts.get(ref['optimal_bitstring_q9_to_q0'],0)==row['aer_best_hits'],(m,p,'optimum hits')
 params=read(E/f'{m}_p{p}_parameters.json');assert len(params['parameters'])==2*p
 assert math.isclose(params['expected_raw_value'],row['expected_raw_value'],abs_tol=1e-8)
 assert row['qiskit_numpy_max_probability_discrepancy']<1e-8,(m,p)
assert len({(r['model'],r['depth_p']) for r in rows})==18
p3=read(R/'evidence/qiskit_aer/stage3_aer_evidence.json')
probs=p3['probabilities_q0_to_q5'];dist=[1.0]
for q in probs:
 nxt=[0.0]*(len(dist)+1)
 for i,t in enumerate(dist):nxt[i]+=t*(1-q);nxt[i+1]+=t*q
 dist=nxt
assert math.isclose(sum(dist[3:]),p3['exact_probability_at_least_three'],abs_tol=1e-12)

# Validate exact decimal probability propagation to Stage 2 economics.
rows=list(csv.DictReader((R/'stage2_candidates.csv').open()))
assert len(rows)==10
s1byid={row['customer_id']:row for row in s1}
for row in rows:
 assert row['predicted_probability']==s1byid[row['customer_id']]['predicted_probability'],('Stage2 probability mismatch',row['customer_id'])
 assert math.isclose(float(row['expected_net_value']),float(row['predicted_probability'])*float(row['expected_deposit_value'])-float(row['contact_cost']),rel_tol=0,abs_tol=1e-10)
# Check that the recorded independent MC uses the exact canonical selected probabilities.
mc=read(R/'evidence/stage3_monte_carlo.json')
assert mc['shots']==8192 and 0<=mc['success_shots']<=8192
assert mc['probabilities_q0_to_q5']==p3['probabilities_q0_to_q5']
assert math.isclose(mc['estimated_probability'],mc['success_shots']/mc['shots'],abs_tol=1e-14)
assert math.isclose(mc['absolute_error'],abs(mc['estimated_probability']-p3['exact_probability_at_least_three']),abs_tol=1e-12)

print('PASS: Stage 1 IDs; Stage 2 210 combinations; Stage 3 exact; all 18 Qiskit Aer count files, parameter lengths and comparison rows')
